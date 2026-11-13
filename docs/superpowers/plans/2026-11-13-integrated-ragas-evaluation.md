# Integrated Supervisor RAGAS Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run RAGAS quality evaluation against the real Supervisor retrieval, answer, and citation pipeline over the final golden set.

**Architecture:** Repair the production Supervisor dependency wiring, then add an evaluation adapter that invokes the compiled Supervisor once per case and converts its state into RAGAS rows. Score the 57 answerable cases with RAGAS, score the 10 REJECT cases structurally, and persist raw and merged reports.

**Tech Stack:** Python 3.12, LangGraph, Neo4j, LangChain OpenAI, RAGAS 0.4.3, pytest.

**Spec:** Current user request in this Codex chat; no standalone spec file.

## Global Constraints

- Do not pass golden-set `expected_facts` or expected IDs into the Supervisor invocation.
- Reuse one Supervisor application and one Neo4j driver for the full run.
- Keep Guardrails graph integration out of scope to avoid conflicting with the Guardrails owner.
- Exclude expected REJECT cases from semantic RAGAS averages and report them separately.

## Review Focus

- Empty or withheld answers must not be reported as successful semantic evaluations.
- Graph evidence must expose original source text rather than only a synthesized path.
- Duplicate evidence and chunk IDs must not inflate contexts or Hit@K.
- A failed case must be checkpointed with its error instead of aborting the full run.
- Evaluation reference values must never enter the production Supervisor input.

---

### Task 1: Repair Supervisor runtime wiring

**Files:**
- Modify: `scripts/run_supervisor.py`
- Test: `tests/test_run_supervisor.py`

**Interfaces:**
- Produces: `build_app()` passing a Neo4j-backed `evidence_lookup` and `LLMSupportChecker` to `build_supervisor()`.

- [ ] Write a failing test proving both dependencies are passed.
- [ ] Run the test and confirm the current missing dependency failure.
- [ ] Add the minimal dependency wiring.
- [ ] Run the test and confirm it passes.

### Task 2: Add the integrated evaluation adapter

**Files:**
- Create: `scripts/evaluation/evaluate_integrated_ragas.py`
- Test: `tests/test_evaluate_integrated_ragas.py`

**Interfaces:**
- Consumes: `build_app()`, golden-set JSONL, processed chunks JSONL.
- Produces: `run_supervisor_cases(...)`, `build_ragas_dataset(...)`, and `merge_ragas_scores(...)`.

- [ ] Write failing tests for state conversion, REJECT handling, deduplication, Hit@K, and per-case error capture.
- [ ] Run the tests and confirm the adapter is missing.
- [ ] Implement the minimal adapter and CLI.
- [ ] Run the tests and confirm they pass.

### Task 3: Verify the runnable evaluation path

**Files:**
- Modify only if verification exposes a defect in Task 1 or Task 2.

**Interfaces:**
- Produces: CLI help and a no-network unit-test path that prove the evaluation can be launched.

- [ ] Run focused tests for the new runtime and evaluator.
- [ ] Run the complete test suite and report any unrelated failures or external-service hangs.
- [ ] Run CLI `--help` and verify required arguments and outputs are documented.
