# Screencast: "Three Routers and a ValueError" (executive-ai-assistant)

Triage an incoming email. Ignore it, notify the owner, answer it, or
schedule the meeting it asks for; let a human approve anything that leaves
the mailbox; survive the human saying no.

## The Original

[`langchain-ai/executive-ai-assistant`](https://github.com/langchain-ai/executive-ai-assistant)
(`eaia/main/graph.py`, 190 lines, 13 nodes) triages with an LLM, has a second
LLM pick a tool (`ResponseEmailDraft`, `MeetingAssistant`,
`SendCalendarInvite`, `Question`, `Ignore`), sends the choice to a human
through Agent Inbox, and routes on the shape of the last message. Three
conditional routers (`route_after_triage`, `take_action`,
`enter_after_human`) each end in `raise ValueError`; four interrupt handlers
(`human_inbox.py`, 395 lines) each re-implement the accept / edit / ignore /
respond branches; a `bad_tool_name` node exists to strip a stray `:` from
tool names the model sometimes emits. Nothing in the graph says *why* an
email is answered or what "handled" means.

## The Contrast

**before.py** — a condensed, hermetic version of that control flow: the same
nodes, the same three routers with their `ValueError` fall-throughs, one
Agent Inbox round trip shared by the four handlers, the `bad_tool_name`
patch loop, a scripted drafter instead of the LLM.

**after.py** — a `FunctionalSensor` triages once; seven `ActionSpec`s carry
the paths as preconditions (`needs_meeting`, `needs_reply`,
`reply_context_ready`, `follow_up_done`); the goal is `email_handled`.
`send_email` is gated with `require_human_approval=True`; `send_cal_invite`
is gated with a Pydantic `CalendarInvite` form whose validated answer lands
in `world_state["invite"]`. No routers.

**after_disrupted.py** — the operator denies the send (the action fails, is
blacklisted, and the planner routes through `escalate_to_user`); the invite
form fails validation (same mechanism); the process dies while the human is
deciding and a new process resumes the thread from the checkpointer.

| | Hand-wired (`before.py`) | GOAP (`after.py`) |
|---|---|---|
| Which path | 3 routers, 11 `if` branches, 4 `raise ValueError` | triage facts as preconditions |
| Human gate | 4 handlers × (accept / edit / ignore / respond) | `require_human_approval=True` or a Pydantic model |
| Human says no | `Ignore` tool call synthesised so a router can see it | action failure, blacklist, re-plan |
| Bad model output | `bad_tool_name` node patches the tool name | no tool names to patch |
| Crash mid-approval | checkpointer (same) | checkpointer (same) |

## Notebook

[`email_assistant_screencast.ipynb`](email_assistant_screencast.ipynb) runs
all three scripts with narration and executed output cells. It is generated
by `_build_notebook.py` so it cannot drift from the scripts:

```bash
uv run python examples/screencast/email_assistant/_build_notebook.py
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace \
    examples/screencast/email_assistant/email_assistant_screencast.ipynb
```

## Run

```bash
cd LangGOAP
uv run python examples/screencast/email_assistant/before.py
uv run python examples/screencast/email_assistant/after.py
uv run python examples/screencast/email_assistant/after_disrupted.py
```

## Test

```bash
uv run pytest tests/integration/test_screencast_email_assistant.py -vv
```
