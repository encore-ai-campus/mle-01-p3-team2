from src.retrievers.answer import Citation, CitedAnswer, Claim, Evidence
from src.supervisor.citation_verifier import AnswerReview, verify_citations


def test_conflicting_pesticide_evidences():
    # 실제 농약 사용정보가 아닌 검증 전용 가상 데이터
    evidence_a = Evidence(
        evidence_id="EVD_CONFLICT_A",
        document_id="DOC_TEST_A",
        title="가상 출처 A",
        url="https://example.com/a",
        quote="작물=배추; 제품=가상약제; 병해충=뿌리혹병; 사용시기=정식전; 희석배수=1000배",
        source="vector",
    )

    evidence_b = Evidence(
        evidence_id="EVD_CONFLICT_B",
        document_id="DOC_TEST_B",
        title="가상 출처 B",
        url="https://example.com/b",
        quote="작물=배추; 제품=가상약제; 병해충=뿌리혹병; 사용시기=정식전; 희석배수=2000배",
        source="vector",
    )

    answer = CitedAnswer(
        answer="가상약제의 희석배수에 대해 서로 다른 기록이 존재합니다.",
        claims=[
            Claim(
                text="가상약제의 희석배수 기록이 서로 충돌합니다.",
                citations=[
                    Citation(
                        evidence_id="EVD_CONFLICT_A",
                        quote="희석배수=1000배",
                    ),
                    Citation(
                        evidence_id="EVD_CONFLICT_B",
                        quote="희석배수=2000배",
                    ),
                ],
            )
        ],
    )

    def mock_support_checker(claim, contexts):
        assert len(contexts) == 2
        assert {c["evidence_id"] for c in contexts} == {
            "EVD_CONFLICT_A",
            "EVD_CONFLICT_B",
        }

        return AnswerReview(
            supported=False,
            reason="동일 조건에서 희석배수가 다릅니다.",
            conflict=True,
            conflict_reason="두 출처의 희석배수가 일치하지 않습니다.",
            conflicting_evidence_ids=[
                "EVD_CONFLICT_A",
                "EVD_CONFLICT_B",
            ],
        )

    result = verify_citations(
        answer=answer,
        evidences=[evidence_a, evidence_b],
        support_checker=mock_support_checker,
    )

    assert result.status == "FAIL"
    assert len(result.conflicts) == 1
    assert result.conflicts[0].evidence_ids == [
        "EVD_CONFLICT_A",
        "EVD_CONFLICT_B",
    ]
    assert result.checked_claims == 1
    assert result.supported_claims == 0

def test_supervisor_withholds_conflict_without_retry(monkeypatch):
    from src.retrievers.search_result import SearchResult
    from src.supervisor.graph import build_supervisor

    calls = []

    import src.supervisor.graph as graph_module

    def mock_input_safety(state):
        return {
            "input_intent": "GENERAL",
            "pesticide_use_requested": False,
        }

    def mock_dialog(state):
        return {
            "dialog_action": "SEARCH",
            "pesticide_use_requested": False,
        }

    monkeypatch.setattr(
        graph_module,
        "input_safety_node",
        mock_input_safety,
    )
    monkeypatch.setattr(
        graph_module,
        "dialog_node",
        mock_dialog,
    )

    # 실제 농약 정보가 아닌 테스트 전용 가상 데이터
    texts = {
        "EVD_CONFLICT_A": (
            "작물=배추; 제품=가상약제; 병해충=뿌리혹병; "
            "사용시기=정식전; 희석배수=1000배"
        ),
        "EVD_CONFLICT_B": (
            "작물=배추; 제품=가상약제; 병해충=뿌리혹병; "
            "사용시기=정식전; 희석배수=2000배"
        ),
    }

    def make_results():
        return [
            SearchResult(
                source="vector",
                text=text,
                evidence_ids=[evidence_id],
                chunk_ids=[f"CHK_{evidence_id}"],
            )
            for evidence_id, text in texts.items()
        ]

    def search_fn(question, **kwargs):
        calls.append("search")
        return make_results()

    def hybrid_search_fn(question, **kwargs):
        calls.append("hybrid")
        return make_results()

    def evidence_lookup(evidence_ids):
        return {
            evidence_id: {
                "evidence_id": evidence_id,
                "document_id": f"DOC_{evidence_id}",
                "title": "가상 충돌 테스트 문서",
                "url": "https://example.com/test",
                "text": texts[evidence_id],
                "chunk_id": f"CHK_{evidence_id}",
            }
            for evidence_id in evidence_ids
        }

    def answer_generator(question, evidences):
        ids = {e.evidence_id for e in evidences}
        assert ids == set(texts), f"근거 전달 실패: {ids}"

        return CitedAnswer(
            answer="가상약제의 사용조건에 대한 기록을 확인했습니다.",
            claims=[
                Claim(
                    text="가상약제의 희석배수는 1000배이다.",
                    citations=[
                        Citation(
                            evidence_id="EVD_CONFLICT_A",
                            quote="희석배수=1000배",
                        ),
                        Citation(
                            evidence_id="EVD_CONFLICT_B",
                            quote="희석배수=2000배",
                        ),
                    ],
                )
            ],
        )

    def conflict_checker(claim, contexts):
        assert len(contexts) == 2

        return AnswerReview(
            supported=False,
            reason="동일 조건에서 값이 상충합니다.",
            conflict=True,
            conflict_reason="가상 근거 A와 B의 희석배수가 다릅니다.",
            conflicting_evidence_ids=[
                "EVD_CONFLICT_A",
                "EVD_CONFLICT_B",
            ],
        )

    def mock_guardrails(state):
        return {
            "guardrail_decision": "ALLOW",
            "guardrail_reason": "통합 테스트 Mock",
            "final_status": state.get("final_status", "ANSWERED"),
            "final_answer": state.get("final_answer", ""),
        }

    app = build_supervisor(
        graph_search_fn=search_fn,
        vector_search_fn=search_fn,
        hybrid_search_fn=hybrid_search_fn,
        evidence_lookup=evidence_lookup,
        cited_answer_generator=answer_generator,
        citation_support_checker=conflict_checker,
        guardrails_fn=mock_guardrails,
        max_retries=1,
    )

    state = app.invoke({
        "question": (
        "배추 뿌리혹병에 대한 가상약제의 등록 자료를 "
        "정보 조회 목적으로 비교해 주세요. "
        "실제 사용이나 살포 방법을 추천하지 말고, "
        "두 자료의 수치가 서로 일치하는지만 확인해 주세요."
    )
    })

    print("\n실제 검색 호출:", calls)
    print("방문 노드:", state.get("visited_nodes"))
    print("입력 의도:", state.get("input_intent"))
    print("Guardrails 판단:", state.get("guardrail_decision"))
    print("Guardrails 사유:", state.get("guardrail_reason"))
    print("최종 답변:", state.get("final_answer"))
    print("경고:", state.get("warnings"))
    print("최종 상태:", state.get("final_status"))
    print("재검색 횟수:", state.get("retry_count"))
    print("인용 검증:", state.get("citation_status"))
    print("충돌 근거:", state.get("citation_conflicts"))
    print("보류 사유:", state.get("withhold_reason"))

    assert state["final_status"] == "WITHHELD"
    assert state.get("retry_count", 0) == 0
    assert len(calls) == 1
    assert len(state.get("citation_conflicts", [])) == 1
    assert "CONFLICT" in state.get("withhold_reason", "")

    conflict = state["citation_conflicts"][0]
    assert set(conflict["evidence_ids"]) == set(texts)

    # 사용자에게 충돌 수치를 임의로 추천하지 않아야 함
    final_answer = state.get("final_answer", "")
    assert "1000배" not in final_answer
    assert "2000배" not in final_answer

