# Simple Streamlit Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unrelated company-information Streamlit entry point with a minimal chat UI that invokes the project's existing GraphRAG Supervisor and displays `final_answer`.

**Architecture:** Keep Supervisor and Neo4j construction in `scripts.run_supervisor.build_app()` and cache its returned resources in Streamlit. Put the small, testable answer-extraction boundary in `src/streamlit/chatbot.py`; keep Streamlit rendering and session-state handling in `src/streamlit/app.py`.

**Tech Stack:** Python 3.12, Streamlit 1.59.1, pytest, existing LangGraph Supervisor and Neo4j runtime

**Spec:** `docs/superpowers/specs/2026-11-13-simple-streamlit-chatbot-design.md`

## Global Constraints

- Reuse `scripts/run_supervisor.py::build_app()` without changing Supervisor search, routing, or answer-generation behavior.
- Provide only message history, question input, answer output, spinner, and conversation reset.
- Persist messages only in Streamlit session state.
- Keep the cached Supervisor and Neo4j driver alive when chat history is reset.
- Do not retain the existing company graph, map, statistics, or evidence visualizations in the active UI.

## Review Focus

- A whitespace-only Supervisor answer must produce the fixed fallback message rather than a blank chat bubble.
- A non-string `final_answer` must produce the same fallback message.
- A Supervisor exception must become a readable Korean error message without crashing the page.
- A question must be sent exactly once as `{"question": question}`.
- Resetting conversation history must not clear the cached Supervisor resource.

---

### Task 1: Testable Supervisor answer boundary

**Files:**
- Replace: `src/streamlit/chatbot.py`
- Create: `tests/test_streamlit_chatbot.py`

**Interfaces:**
- Consumes: an object exposing `invoke(input: dict[str, str]) -> dict[str, object]`
- Produces: `generate_answer(supervisor: object, question: str) -> str`

- [ ] **Step 1: Write failing tests for answer generation**

  Add focused tests asserting exact invocation input, successful `final_answer`, fallback behavior for missing/blank/non-string answers, and a Korean user-facing error for raised exceptions.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run: `uv run pytest tests/test_streamlit_chatbot.py -v`

  Expected: FAIL because the new minimal `generate_answer` interface does not exist.

- [ ] **Step 3: Implement the minimal answer boundary**

  Replace the unrelated company-agent helpers with `generate_answer(supervisor: object, question: str) -> str`. Invoke the Supervisor exactly once, return stripped `final_answer`, use `"답변을 생성하지 못했습니다. 다시 질문해 주세요."` for invalid answers, and convert exceptions to `"답변 생성 중 오류가 발생했습니다: {exc}"`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

  Run: `uv run pytest tests/test_streamlit_chatbot.py -v`

  Expected: all focused tests PASS.

### Task 2: Minimal Streamlit chat entry point

**Files:**
- Replace: `src/streamlit/app.py`
- Verify: `src/streamlit/chatbot.py`

**Interfaces:**
- Consumes: `scripts.run_supervisor.build_app() -> tuple[supervisor, driver]` and `generate_answer(supervisor, question) -> str`
- Produces: Streamlit entry point `main() -> None`

- [ ] **Step 1: Add a failing source-contract test**

  Extend `tests/test_streamlit_chatbot.py` to assert the entry point imports `build_app`, caches the resource, invokes `generate_answer`, uses `st.chat_input`, stores `st.session_state.messages`, and offers a reset button without cache clearing.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run: `uv run pytest tests/test_streamlit_chatbot.py -v`

  Expected: FAIL because the current entry point is the company-information application.

- [ ] **Step 3: Implement the minimal Streamlit page**

  Configure a simple agricultural Q&A page, cache `build_app()` with `st.cache_resource`, render history with `st.chat_message`, accept one `st.chat_input`, show a spinner during `generate_answer`, append both messages, and reset only `st.session_state.messages` from the sidebar.

- [ ] **Step 4: Run focused tests and compile checks**

  Run: `uv run pytest tests/test_streamlit_chatbot.py -v`

  Run: `uv run python -m py_compile src/streamlit/app.py src/streamlit/chatbot.py`

  Expected: tests PASS and compilation exits with status 0.

### Task 3: Regression and startup verification

**Files:**
- Verify: all changed files

**Interfaces:**
- Consumes: completed Tasks 1–2
- Produces: verification evidence only

- [ ] **Step 1: Run the full test suite**

  Run: `uv run pytest -q`

  Expected: all project tests PASS; report any pre-existing failures by test name.

- [ ] **Step 2: Run a bounded headless Streamlit startup smoke test**

  Start `uv run streamlit run src/streamlit/app.py --server.headless true` with a short timeout and verify that the server reaches its ready state. This checks import and page startup only; it does not require submitting a live Neo4j question.

- [ ] **Step 3: Review the diff for scope**

  Confirm the active UI contains no company-specific code and that Supervisor internals were not changed.

