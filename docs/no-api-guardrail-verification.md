# API 없이 가드레일 수정을 확인하는 방법

RAGAS 통합 평가는 답변 생성(Answer Agent), 근거 판정(LLM-as-Judge),
채점(RAGAS 6지표)에 모두 LLM을 쓴다. 67건 전체를 돌리면 수정 하나를
확인할 때마다 비용이 든다. 이 문서는 **비용 0원으로 확인할 수 있는
범위**와 **꼭 LLM이 필요한 범위**를 나누고, 후자의 비용을 줄이는
방법을 적는다.

## 1단계 — 비용 0원: 단위 테스트

```bash
uv run python -m pytest tests/ -q --ignore=tests/test_nemo_dialog_live.py --ignore=tests/test_neo4j_execution_integration.py
```

LLM을 쓰는 경로는 모두 스텁으로 대체돼 있다. 특히
`tests/test_guardrail_validators.py`의 judge 테스트는 OpenAI
클라이언트를 가짜 객체로 바꿔, severity 판정이 보류/통과로 어떻게
이어지는지를 API 없이 검사한다.

## 2단계 — 비용 0원: 저장된 근거로 Output Rail 재실행

이미 저장된 평가 결과의 `retrieved_contexts`는 Neo4j 등록정보를
`키=값; 키=값` 형태로 담고 있다. 이 문자열을 레코드로 되돌리면
농약 사용조건 검증을 그대로 다시 돌릴 수 있다.

```bash
uv run python -m scripts.evaluation.replay_usage_rail_offline
```

케이스마다 세 가지를 검사한다.

| 검사 | 만드는 답변 | 기대 |
|---|---|---|
| 근거 그대로 인용 | 각 제품의 등록값을 그대로 인용 | 통과 |
| 제품 간 수치 혼입 | A 제품 이름 + B 제품 희석배수 | BLOCK |
| 사용횟수 상한 초과 | 등록 상한 + 1회 | BLOCK |

특정 케이스만 보려면:

```bash
uv run python -m scripts.evaluation.replay_usage_rail_offline --case-ids GS018,GS026,GS027,GS058,GS059,GS061
```

### v3 보류 6건에 적용한 결과

| 케이스 | v3 저장된 상태 | v3 코드로 재실행 | 수정 후 |
|---|---|---|---|
| GS018 | BLOCKED | 근거 그대로 인용 10건 중 10건 보류 | 0건 |
| GS026 | WITHHELD | 10건 중 10건 보류 | 0건 |
| GS027 | WITHHELD | 10건 중 10건 보류 | 0건 |
| GS058 | WITHHELD | 9건 중 9건 보류 | 0건 |
| GS059 | WITHHELD | 10건 중 10건 보류 | 0건 |
| GS061 | WITHHELD | 10건 중 10건 보류 | 0건 |

보류 사유는 전부 `희석배수`였다. 등록정보의 `dilution`이 단위 없는
`"1000"`인데 참조 추출이 `"배"`를 요구해, **정답을 써도 보류되는
구조**였다. 근거가 있는 전체 25건에서도 수정 후 기대와 다른 케이스는
0건이다.

## 3단계 — 비용 0원: 검증상태 안내 확인

LLM-as-Judge가 보류한 15건 중 다수는 "검증상태 REVIEW_REQUIRED인
품목을 사용 가능하다고 단정했다"였다. 이 상태 표기는 등록정보에 이미
있는 사실이므로 모델에 맡기지 않고
`src/guardrails/verification_notice.py`가 규칙으로 덧붙인다.

저장된 근거로 안내가 실제로 붙는지 확인할 수 있다.

```bash
uv run python -m pytest tests/test_verification_notice.py -q
```

## LLM이 꼭 필요한 범위

다음 두 가지는 오프라인으로 확인할 수 없다.

1. **Answer Agent 프롬프트 변경** — 제품별 수치 인용, 검증상태 반영
   규칙을 모델이 실제로 지키는지
2. **LLM-as-Judge severity 판정** — 표현 차이를 `WORDING`으로
   분류하는지

### 비용을 줄여 확인하는 순서

수정에 걸린 케이스만 다시 돌린다. 채점 단계를 빼면 LLM 호출은
케이스당 답변 생성 + 판정 2회로 끝난다.

```bash
uv run python -m scripts.evaluation.evaluate_integrated_ragas --stage collect --output-dir data/evaluation/ragas_result/subset --case-ids GS012,GS014,GS018,GS019,GS020,GS021,GS023,GS026,GS027,GS032,GS035,GS036,GS040,GS041,GS054,GS058,GS059,GS061,GS062,GS066,GS067
```

- `--stage collect`는 RAGAS 채점(가장 비싼 단계)을 건너뛴다.
  `final_status`와 보류 사유만 보면 수정 효과를 알 수 있다.
- `--case-ids`로 67건 중 21건만 돌린다.
- `--output-dir`를 따로 주지 않으면 기존 결과 파일을 덮어쓴다.
- 판정 모델은 `GUARDRAIL_JUDGE_MODEL` 환경변수로 바꿀 수 있다
  (기본 `gpt-4o-mini`).

상태가 기대대로 바뀐 것을 확인한 뒤에 전체 RAGAS를 한 번만 돌려
v4를 만든다. 채점 모델은 `--model`로 낮출 수 있다.

## 진단 필드

v3 결과만으로는 보류된 답변의 원문을 알 수 없어, 어떤 주장이 걸렸는지
재실행 없이는 확인할 수 없었다. 평가 레코드에 다음을 추가했다.

- `draft_answer` — 가드레일 적용 전 생성 답변
- `judge_severity` — LLM-as-Judge가 매긴 severity

다음 실행부터는 보류 원인을 결과 파일만 보고 분류할 수 있다.

## 검증 역할 분담

21건 재실행에서 judge가 지적한 수치 불일치 3건(GS012·GS035·GS066)을
공식 등록정보와 대조해보니 **세 건 모두 답변이 맞고 judge가 틀렸다**.
근거 줄이 서로 비슷해(같은 품목이 상표별로 10줄 이상) judge가 다른
제품의 줄과 비교한 것이다. `"2000배인데 답변에서는 2000배로 잘못
기재됨"`처럼 자기모순인 판정 사유도 나왔다.

그래서 판정 책임을 이렇게 나눴다.

| 대상 | 담당 | 근거 |
|---|---|---|
| 희석배수·사용횟수·수확 전 사용시기 | 결정론 레일 | Neo4j 등록정보 전건 |
| 상표 ↔ 등록번호 일치 | 결정론 레일 | 등록번호로 제품 특정 |
| 검증상태·최신 등록 안내 | 규칙 삽입 | `verification_notice.py` |
| 근거에 없는 사실, 대상 혼동, 일반화, 사용 가능 단정 | LLM-as-Judge | 검색 근거 |

judge 프롬프트에서 수치·등록번호 대조와 검증상태 판단을 빼고,
답변이 다루는 범위를 명시했다. 규칙이 21개까지 늘면서 서로 모순됐던
것도 함께 정리했다.

### 대조 비율 측정

수치 판정을 결정론 레일에 넘긴 것이 검증을 느슨하게 만든 게 아닌지
확인해야 한다. 답변의 수치가 실제로 등록정보와 대조되는지 센다.

```bash
uv run python -m scripts.evaluation.measure_usage_verification
```

21건 draft 기준 **285항목 중 283항목(99.3%)이 공식 등록정보와
대조되고, 불일치는 0건**이다. 미대조 2항목은 사용조건 충돌로 목록에서
제외된 제품을 답변이 인용한 경우이며, 차단이 아니라 경고로 전달된다.

judge가 맞힌 수치 불일치는 3건 중 0건이었으므로, 이 교체는 완화가
아니라 강화다.

## 판정 범위를 프롬프트가 아니라 코드로 강제한 이유

judge 프롬프트에 "수치와 검증상태는 다른 검증이 담당하니 다루지
말라"고 적었더니, 21건 재실행에서 보류 13건이 **전부** 그 범위 밖
사유였다. UNSUPPORTED가 사라진 대신 같은 이견이 CONFLICT로 이름만
바꿔 돌아왔다.

| | 2차 실행 | 3차 실행(안내문만) |
|---|---|---|
| ANSWERED | 2 | 8 |
| 보류 | 19 | 13 |
| 보류 중 범위 밖 사유 | — | **13/13** |

그래서 사유를 자유 문장이 아니라 분류로 받고, 범위 밖 분류는 코드가
보류로 쓰지 않도록 했다.

| category | 처리 |
|---|---|
| `FABRICATED`, `WRONG_TARGET`, `OVERGENERALIZED`, `USABILITY_CLAIM` | 보류 |
| `REGISTERED_VALUE`, `VERIFICATION_STATUS`, `COMPLETENESS`, `WORDING`, `NONE` | 경고로 전달하고 답변 |
| 분류 없음·알 수 없는 값 | 보류 (안전한 쪽) |

판정 모델에게 침묵을 요구하는 대신 **이견을 적을 칸을 주고 코드가
걸러낸다.** 분류를 틀리게 적으면(수치 이견을 `FABRICATED`로) 여전히
보류되므로, 평가 레코드에 `judge_category`를 남겨 다음 실행에서
준수 여부를 바로 볼 수 있게 했다.

사용조건 검증을 근거 일치 판정보다 **먼저** 돌리고, 대조를 마친
항목을 판정 모델에 `[이미 확인된 사항]`으로 함께 넘긴다.
