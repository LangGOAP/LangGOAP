# Security Policy

## Overview

LangGOAP values the work of the security community and welcomes
responsible disclosure of potential security vulnerabilities. Good-
faith research helps us keep the project, its users, and the wider
LangChain ecosystem safe.

LangGOAP is a pure-Python library. It runs inside the host process of
whatever application embeds it, so almost all realistic threats
materialise when LangGOAP is given untrusted input — untrusted natural-
language goals, untrusted action bodies, untrusted records read from a
`BaseStore`, and so on. This policy focuses on that attack surface.

## Supported versions

LangGOAP is pre-1.0. Security fixes are backported only to the most
recent patch release of the current minor series. Older releases do
not receive security updates.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅ |
| < 0.1.0 | ❌ |

## Scope

### What's in scope

- Remote code execution paths reachable through:
  - `ActionSpec.execute` and `ActionSpec.aexecute`,
  - `ActionSpec.effect_validator`,
  - the `GoalInterpreter` LLM pipeline when parsing attacker-controlled
    natural-language goals,
  - the `GoapSubgraph` / `create_goap_agent` / `goapify_tool`
    integration surfaces when wired to untrusted LangChain tools.
- Deserialization issues in `StoreExecutionHistory` when reading
  attacker-influenced records from a LangGraph `BaseStore`.
- Planner denial-of-service vectors (for example, inputs that cause
  unbounded A* expansion, runaway CP-SAT search, or pathological
  blacklist feedback loops) **when the inputs are plausibly untrusted**
  — NL goals from end users, world-state snapshots from untrusted
  checkpointers, action catalogs assembled from user-supplied tool
  descriptions.
- Information disclosure through tracing hooks, execution history, or
  visualization output leaking data the caller did not intend to
  expose.

### What's out of scope

Always out of scope:

- Social engineering or phishing targeting LangGOAP maintainers or
  contributors.
- Physical attacks or infrastructure access.
- Attacks against third-party services (LangSmith, OpenAI, Anthropic,
  Google OR-Tools build infrastructure, and so on) — report those
  upstream.
- Issues that only affect LangGOAP users without a LangGOAP-controlled
  vulnerability (for example, a user who pipes secrets into
  `ActionSpec.metadata` and then prints them).

Usually out of scope unless additional, concrete impact is demonstrated:

- Automated scanner output, indiscriminate fuzzing, or LLM-generated
  vulnerability reports with no working proof of concept.
- Known-vulnerable transitive dependencies without a LangGOAP-specific
  reachable exploit path — report upstream to `langgraph`,
  `langchain-core`, `ortools`, and so on.
- Prompt injection against the default `GoalInterpreter` prompt
  template without a working exploit that changes the planner's
  behaviour in a way the caller cannot recover from.
- Issues that require a malicious local admin or write access to the
  host running LangGOAP.
- General performance regressions in the A*/CSP pipeline that do not
  come from adversarial input.
- Missing security headers, TLS, or CSP on `integrallis.github.io/langgoap`
  (that's GitHub Pages infrastructure, not LangGOAP code).

## How to report a vulnerability

**Do not open a public GitHub issue** for security reports.

Please report suspected vulnerabilities privately via
[GitHub Security Advisories](https://github.com/LangGOAP/LangGOAP/security/advisories/new).
If that channel is unavailable to you, email
`security@integrallis.com` with the subject line
`[LangGOAP security] <short description>`.

## Reporting guidelines

### What to include

Reports should contain enough detail for us to reproduce and assess
the issue. At a minimum please provide:

- A clear description of the vulnerability.
- The affected LangGOAP version(s), and if possible the commit SHA.
- A minimal, self-contained reproducer — a Python script, test, or
  notebook a maintainer can copy and run as-is.
- Steps to reproduce, including any environment prerequisites.
- An explanation of the security impact, grounded in realistic
  deployments.
- Relevant logs, stack traces, or screenshots (if applicable).

Reports that lack sufficient detail to reproduce the issue may be
closed without remediation.

### Proof of impact

We prioritise reports that clearly demonstrate **realistic security
impact**. Where possible, show:

- How the issue could be exploited in practice.
- What an attacker could gain (code execution, data exfiltration,
  planner denial-of-service against a long-running agent, and so on).
- Any constraints or prerequisites required to exploit.

Theoretical issues or best-practice gaps without demonstrated impact
are generally out of scope.

### Testing expectations

Please conduct testing responsibly:

- Only test against local checkouts or installs you control.
- Do not access, modify, or delete data that does not belong to you.
- Do not intentionally degrade availability of any shared service.
- Stop testing immediately if you believe your actions could impact
  other users or production systems.

### Submission rules

- Submit **one vulnerability per report** unless chaining is required
  to demonstrate impact.
- Duplicate submissions are resolved based on the first reproducible
  report received.
- Vulnerabilities discovered through automated scanning must include
  manual validation and a demonstrated impact narrative.
- We do **not** accept AI-generated submissions or reports authored
  primarily by automated tools.

## Response targets

LangGOAP is maintained by a small team. We make a best-effort attempt
to meet the following timelines:

- **Initial acknowledgement:** within 3 working days.
- **Initial triage / severity assessment:** within 10 working days.
- **Coordinated fix published:** varies with severity and complexity.

Timelines are best-effort and may vary with report quality, severity,
and volume.

## Disclosure expectations

- **Do not publicly disclose vulnerabilities without the maintainers'
  explicit written permission.**
- Allow reasonable time for investigation and remediation before any
  public write-up.
- Coordinated disclosure may be permitted after remediation at the
  maintainers' discretion. Reporters will be credited in release
  notes unless they request otherwise.

## Safe harbor

Security research conducted **in good faith and in accordance with
this policy** is considered authorised.

The LangGOAP maintainers will not initiate legal action against
researchers who comply with this policy. If a third party initiates
legal action related to compliant research, we will make reasonable
efforts to clarify that the activity was authorised.

## Other security concerns

For any non-vulnerability security questions (for example, threat
modelling advice for an application that embeds LangGOAP), please
contact `security@integrallis.com`.
