# Changelog

All notable changes to LangGOAP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — Unreleased

Initial release. LangGOAP is a Goal-Oriented Action Planning framework
for LangGraph with constraint optimization, natural-language goal
interpretation, and a LangChain-first execution model in which the plan
*is* a compiled `StateGraph`.

`0.1.0` was uploaded to TestPyPI with a broken console script (`click`
sat under the `[cli]` extra while the `langgoap` entry point was
always registered) and has been yanked.  `0.1.1` is the first
installable release; no public API or behavior differs from the
`0.1.0` source tree.
