@'
# Golden Set Contract

## 1. 목적
본 문서는 농업 GraphRAG의 Router 및 Retriever 평가에 사용하는 Golden Set의 데이터 구조와 검증 규칙을 정의한다.

## 2. 데이터 구성
- 전체: 67개
- GRAPH: 21개
- VECTOR: 19개
- HYBRID: 17개
- REJECT: 10개
- 확정 데이터의 status는 READY여야 한다.
- case_id는 고유해야 한다.
- 동일한 정답 근거 ID 조합의 중복 사례를 허용하지 않는다.

## 3. 질문 규칙
- question은 비어 있을 수 없다.
- 지시 대상이 불분명한 지시어는 허용하지 않는다.
- 지시어를 사용하는 경우 질문 내부에 식별 가능한 선행 대상이 있어야 한다.

## 4. 예상 답변 행동
- ANSWER: 일반적인 근거 기반 답변
- ANSWER_WITH_WARNING: 안전 관련 주의사항을 포함하는 답변
- REJECT: 서비스 범위 밖 질문에 대한 거절

농약 사용조건, 비료 표준사용량, 검증상태 미확정, 품종 저항성 추정 등이 포함되면 경고 필요성을 검사한다.

ANSWER_WITH_WARNING에는 safety_assertions가 필요하다.
경고가 필요하지 않은 일반 답변에도 불필요한 안전성 주장을 추가해서는 안 된다.

## 5. 검색 전략 및 구조
- VECTOR 사례에는 그래프 관계 ID를 지정하지 않는다.
- GRAPH 사례는 단일 홉 경로 규칙을 따른다.
- HYBRID 사례는 2홉 경로 규칙을 따른다.
- GRAPH 사례에는 필요한 엔티티 및 관계 정보가 포함되어야 한다.
- Evidence, Entity, Relation 및 Source Chunk ID는 중복 없이 관리한다.
- 코퍼스 검증 시 참조 ID는 실제 데이터에 존재해야 한다.

## 6. REJECT 규칙
- query_strategy는 reject여야 한다.
- expected_behavior는 REJECT여야 한다.
- reference 및 정답 근거 관련 필드는 비어 있어야 한다.

## 7. 검증
검증 구현:
scripts/evaluation/validate_golden_set.py

테스트 구현:
tests/test_golden_set_contract.py

Golden Set의 스키마, ID 참조, 경고 규칙 및 검색 경로 구조를 자동 검증한다.
'@ | Set-Content docs/golden-set-contract.md -Encoding UTF8

uv run python -m pytest tests/test_golden_set_contract.py -q