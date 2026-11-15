"""Generate ``weather_alert_screencast.ipynb`` (run from the repo root).

    uv run python examples/screencast/weather_alert/_build_notebook.py
    uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace \
        examples/screencast/weather_alert/weather_alert_screencast.ipynb

Markdown cells carry the narration; code cells import the three scripts so
the notebook and the scripts cannot drift apart. Everything is hermetic.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "weather_alert_screencast.ipynb"
S = nbf.v4.new_markdown_cell
C = nbf.v4.new_code_cell

cells: list = []

cells.append(S("""# Weather disaster management — before / after / after-disrupted

Notebook version of the three-script screencast in
[`examples/screencast/weather_alert/`](https://github.com/LangGOAP/LangGOAP/tree/main/examples/screencast/weather_alert).

**The task.** Grade a forecast, brief the right department, get a human to sign off on mild alerts,
send the alert. Survive the relay and the operator saying no.

**The original.** [`NirDiamant/GenAI_Agents`](https://github.com/NirDiamant/GenAI_Agents) →
[`Weather_Disaster_Management_AI_AGENT.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/Weather_Disaster_Management_AI_AGENT.ipynb),
Part 1. It fetches OpenWeatherMap data, asks a local LLM to name a disaster type and grade its
severity, routes to one department with `route_response`, asks a human (blocking `input()`) for
low/medium alerts and auto-approves the rest, then sends an e-mail or runs `handle_no_approval`.

Three things the routing hides, all reproduced below:

- `route_response`'s first branch is `return "emergency_response"` followed by a bare
  `"send_email_alert"` string. A second target was evidently intended; it is unreachable.
- `get_human_verification` auto-approves anything that is not `"low"` or `"medium"`, so a hurricane
  whose severity call failed is routed to civil defense and e-mailed with no human in the loop.
- `send_email_alert` wraps `smtplib` in `try/except` and the graph ends as if the alert had gone out.

| | Hand-wired (`before.py`) | GOAP (`after.py`) |
|---|---|---|
| Which department | `route_response` string matching, one dead branch | preconditions on `severe` / `flood_or_storm` |
| Who needs a human | `if severity in ["low", "medium"]` inside a node | `approve_alert` with precondition `severe: False` |
| Human says no | `verify_approval_router` → `handle_no_approval` | action failure → stuck handler relaxes the goal → `withhold_alert` |
| Relay down | `try/except`, message appended, graph ends | `ActionQos` retry, then blacklist and SMS fallback |
| No threat | routed to civil defense and e-mailed | `log_all_clear` |
"""))

cells.append(S("""## 1. The hand-wired version

`before.py` keeps the nodes, the edges and both routers verbatim. The weather API, the LLM, `input()`
and SMTP are fixtures and flags. Read `route_response` first."""))

cells.append(C("""import inspect

from examples.screencast.weather_alert import before

print(inspect.getsource(before.route_response))
print(inspect.getsource(before.verify_approval_router))"""))

cells.append(C("""print(before.build_graph().compile().get_graph().draw_mermaid())"""))

cells.append(C("""for label in ("hurricane", "flood", "heatwave", "calm"):
    print(f"--- {label} ---")
    final = before.run(label)
    print(f"  alerts={final['alerts']} human_asked={final['human_asked']}\\n")"""))

cells.append(
    S(
        """The calm forecast was routed to civil defense and e-mailed: `route_response` has no branch for
"no immediate threat". Now the three failure modes."""
    )
)

cells.append(C("""print("--- operator rejects the flood alert ---")
final = before.run("flood", approve=False)
print(f"  alerts={final['alerts']}\\n  {final['messages'][-1].content}\\n")

print("--- the severity LLM call fails during a hurricane ---")
final = before.run("hurricane", severity="Assessment Failed")
print(f"  routed to {final['department']}; human_asked={final['human_asked']}; alerts={final['alerts']}\\n")

print("--- SMTP is down during a hurricane ---")
final = before.run("hurricane", smtp_down=True)
print(f"  alerts={final['alerts']}\\n  {final['messages'][-1].content}")"""))

cells.append(S("""## 2. The GOAP version: the forecast is a set of facts

A `FunctionalSensor` grades the forecast once and writes `threat`, `severe`, `flood_or_storm`, and
`approved` when the event is severe. That last line *is* the auto-approval policy. Eight actions carry
the rest as preconditions; the goal is `forecast_handled`."""))

cells.append(
    C(
        """from examples.screencast.weather_alert.after import (
    SCENARIOS, FakeSmtp, assess, build_actions, build_goal, build_graph, initial_world_state,
)

for label, weather in SCENARIOS.items():
    print(f"{label:10s} ->", assess({"weather": weather}))
print()
for a in build_actions():
    extras = ("  [approve/deny]" if a.require_human_approval else "") + (f"  qos={a.qos.max_attempts} attempts" if a.qos else "")
    print(f"{a.name:22s} cost={a.cost:<4} pre={dict(a.preconditions)}  eff={dict(a.effects)}{extras}")"""
    )
)

cells.append(
    S(
        """The same four forecasts. The hurricane is severe, so no human is asked. The flood and the heatwave
pause at `approve_alert`. The calm forecast runs `log_all_clear` and sends nothing."""
    )
)

cells.append(C("""from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command


def handle(label: str, resume=True) -> dict:
    graph = build_graph()
    compiled = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": label}}
    print(f"--- {label}: {SCENARIOS[label]['city']} ---")
    result = compiled.invoke(
        graph.initial_state(build_goal(), initial_world_state(SCENARIOS[label])), config=config
    )
    if "__interrupt__" in result:
        ask = result["__interrupt__"][0].value
        print(f"  [human] {ask['type']} for {ask['action']} -> {resume}")
        result = compiled.invoke(Command(resume=resume), config=config)
    print(f"  status: {result['status']}  steps: {[h['action_name'] for h in result['execution_history']]}\\n")
    return result


for label in SCENARIOS:
    handle(label)"""))

cells.append(
    S("""## 3. Disruption: the operator says no, the relay flaps, the relay dies

A denial fails `approve_alert` with the operator's reason attached. The observer blacklists it, and
now no action produces `approved`, so the planner has no path to `forecast_handled`. That is where the
`FunctionalStuckHandler` runs: it relaxes the goal to `decision_logged` and records the reason, and
`withhold_alert` is the plan. The run ends `goal_achieved` against the relaxed goal, which is visible in
the result.""")
)

cells.append(C("""import logging, sys

logging.basicConfig(stream=sys.stdout, format="  log  %(message)s", level=logging.WARNING, force=True)

from examples.screencast.weather_alert import after_disrupted

denied = after_disrupted.denied_alert()"""))

cells.append(
    S(
        """A dropped SMTP connection is retried inside the action by `ActionQos` (no re-plan, one history entry).
A relay that rejects the message outright is not retryable: the action fails, is blacklisted, and
`send_sms_alert` (cost 3) is the next cheapest route to the same effect."""
    )
)

cells.append(C("""flaky = after_disrupted.flaky_smtp()
dead = after_disrupted.dead_smtp()"""))

cells.append(
    S("""## 4. What replaced what

| Hand-wired piece | GOAP replacement |
|---|---|
| `route_response` (3 branches, 1 unreachable literal) | preconditions on `severe` / `flood_or_storm` / `threat` |
| `get_human_verification` (`input()` loop + auto-approve `else`) | `approve_alert` with `require_human_approval=True`; `approved: True` from the sensor when severe |
| `verify_approval_router` + `handle_no_approval` | blacklist, stuck handler, `withhold_alert` |
| `try/except` around `smtplib` | `ActionQos(max_attempts=3, retryable_exceptions=(ConnectionError,))`, then `send_sms_alert` |
| "No immediate threat" → civil defense → e-mail | `log_all_clear` |

The integration test
[`tests/integration/test_screencast_weather_alert.py`](https://github.com/LangGOAP/LangGOAP/blob/main/tests/integration/test_screencast_weather_alert.py)
pins the four paths, the gate payload, the denial-and-relax path, both relay failures and the async
path, and keeps the hand-wired version faithful to upstream (dead literal included).""")
)


def main() -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, str(OUT))
    print(f"wrote {OUT.relative_to(HERE.parents[2])}")


if __name__ == "__main__":
    main()
