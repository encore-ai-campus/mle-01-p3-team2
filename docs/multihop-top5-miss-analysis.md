# Multi-hop 검색 실패 16건 원인 분리 분석

대상: `data/evaluation/ragas_result/ragas_results.jsonl`(2026-11-14 실행)에서 `question_type=multi_hop`
19문항 중 `chunk_hit_at_k=0`인 **16문항** — GS011, GS012, GS018, GS020, GS023, GS026, GS027,
GS035, GS036, GS054, GS057, GS059, GS061, GS062, GS066, GS067.

재현 스크립트: `scripts/evaluation/diagnose_multihop_top5_misses.py`
산출물: `data/evaluation/multihop_top5_miss_diagnosis.jsonl`
Gold ID는 검색 입력·점수 계산에 쓰지 않고 사후 순위 확인에만 사용했다.

## 1. 결론 요약

| 질문 | 답 |
|---|---|
| 정답 2-hop 관계·엔티티가 후보에 있었나 (현 운영 설정) | **0/16. 후보에 아예 없다.** 두 번째 hop이 한 번도 탐색되지 않았다. |
| Top-5에서 밀린 것인가 | **경로 단위로는 아니다.** 단, hop-1 근거 chunk 단위로는 13/16이 후보에 있고 순위에서 밀렸다. |
| 그래프에 Gold 경로가 없는 건가 | **아니다. 15/15 모두 Neo4j에 정방향 2-hop으로 존재하고, 설정만 바꾸면 전부 회수된다.** |

따라서 이 16건은 "랭킹이 아쉬운 문제"가 아니라 **후보 생성 단계에서 2-hop 경로가 만들어지지
않는 설정 문제**다. 라우터가 내보내는 `max_hops`와 `relation_types`가 `GraphRetriever.search`에서
사실상 하드 필터로 작동해, 2-hop 탐색이 빈 결과를 내고 1-hop 폴백으로 떨어진다.

## 2. 측정 방법

문항별로 네 가지 설정을 돌려 Gold 순위를 비교했다.

| 설정 | 내용 | 의미 |
|---|---|---|
| A (현 운영) | 라우터가 준 `max_hops`·`relation_types`, `limit=10` | 파이프라인과 동일 |
| B | `max_hops=2`, 관계 타입 필터 없음, `limit=10` | hop 수·관계 필터만 교정 |
| C | B + `limit=30`, `candidate_pool_size=100` | 후보 풀까지 확대 |
| D | `max_hops=2`, 필터 없음, `candidate_pool_size=166`(fetch 498) | 회수 가능성 상한 |

## 3. 핵심 관측

### 3.1 현 설정에서는 2-hop 경로 자체가 생성되지 않는다

16문항의 Gold 경로는 **전부** `Crop -[HAS_REGISTERED_PESTICIDE_USE]-> PesticideUse
-[CONTROLS_TARGET]-> Disease/Insect` 형태다(Neo4j 직접 확인).

- 설정 A에서 반환된 결과의 hop 수: **14/16 문항이 전부 1-hop**, GS011은 **결과 0건**,
  GS020은 VECTOR로 라우팅되어 그래프 검색 미실행.
- 그 결과 Gold 2번째 관계(`CONTROLS_TARGET`)와 3번째 엔티티(`DIS_*`/`INS_*`)의 후보 내 순위는
  **16문항 전부 없음(None)**. Top-200까지 넓혀도 없다.
- 반면 Gold 1번째 엔티티(작물)는 16/16 전부 1위에 있고, 1번째 관계와 Gold chunk는 hop-1 결과에
  붙어 있어 후보 풀을 넓히면 나타난다.

원인은 라우터 출력과 `graph_retriever.search`의 상호작용이다.

1. **`max_hops=1` 오분류 6건** — GS011, GS018, GS023, GS026, GS059, GS062.
   `search`의 2-hop 우선 탐색 분기는 `max_hops >= 2` 조건이라 아예 진입하지 못한다.
2. **관계 타입 조합 오류 14/16** — 라우터가 2번째 타입으로 `AFFECTED_BY_INSECT` /
   `AFFECTED_BY_DISEASE` / `HAS_ENVIRONMENT_REQUIREMENT`를 내보낸다. 이들은 `Crop`에서 출발하는
   간선이라 `PesticideUse` 뒤에 이어지지 않는다. `{HAS_REGISTERED_PESTICIDE_USE, AFFECTED_BY_*}`로
   2-hop을 찾으면 유효 경로가 0건이고, 같은 타입으로 재시도하는 broad 탐색도 0건이어서
   마지막 `min_hops=1` 폴백 결과(1-hop)가 그대로 반환된다. `CONTROLS_TARGET`을 포함한 경우는
   GS011·GS026 2건뿐인데, 두 건 모두 `max_hops=1`이라 역시 2-hop에 도달하지 못한다.
3. **GS011은 결과 0건** — 라우터가 `max_hops=1` + `relation_types=['CONTROLS_TARGET']`을 주고,
   `Crop`에는 `CONTROLS_TARGET` 간선이 없어 1-hop 탐색이 공집합이 된다. 앵커링은 정상
   (`entity_match_count=1`, 시작 엔티티 `CRP_A18CD4C77384F16E` 매칭)이었다.
4. **GS020은 라우팅 실패** — 기대 전략 hybrid인데 VECTOR로 라우팅되어 그래프 결과 0건이고,
   Gold chunk는 벡터 Top-200에도 없다. 유일한 "검색 경로 + 벡터 회수" 동시 실패 사례.

### 3.2 chunk 단위로는 "순위에서 밀린" 문제가 맞다

hop-1 관계에 Gold chunk가 붙어 있어, 후보 풀을 넓히면 13/16에서 Gold chunk가 보인다.
다만 순위가 3위~186위로 흩어져 Top-5에 들어오지 못한다. 동일 작물·동일 방제 대상에 연결된
`PesticideUse` 형제 경로가 **46~257개**라 제품 식별 단서가 없는 질문에서는 Gold 1건을
위로 올릴 신호가 없다.

### 3.3 설정만 바꾸면 Gold 경로는 전부 회수된다

설정 D(2-hop + 관계 필터 제거 + 후보 풀 166)에서 Gold 2-hop 경로 회수율은 **15/15**
(GS057은 Gold 엔티티·관계가 비어 있는 vector 전략 문항이라 제외). 즉 데이터·그래프 문제는 없다.

설정 C(`limit=30`, 풀 100) 기준으로는 10/15가 Top-30 안에 들어오고, 설정 B(`limit=10`)에서는
4건(GS011 2위, GS054 1위, GS059 8위, GS061 1위)이 Top-10에 들어온다.

## 4. 문항별 표

`pipe`=현 설정 Gold chunk 순위, `deep`=후보 풀 확대 시 Gold chunk 순위,
`C/D`=2-hop 교정 후 Gold **경로** 순위, `형제`=동형 경쟁 경로 수.

| 문항 | 라우팅 | router max_hops | router 2번째 관계 | pipe chunk | deep chunk | C 경로 | D 경로 | 형제 | 1차 원인 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| GS011 | GRAPH | 1 | – (`CONTROLS_TARGET` 단독) | – | – | 3 | 3 | 4 | 관계 필터로 결과 0건 |
| GS012 | HYBRID | 2 | AFFECTED_BY_INSECT | – | 3 | 2 | 3 | 245 | 2-hop 미탐색 + 순위 경쟁 |
| GS018 | GRAPH | 1 | – | – | 77 | – | 135 | 200 | max_hops=1, 경쟁 심함 |
| GS020 | VECTOR | 2 | – | – | – | 4 | 92 | 117 | 라우팅 오류(+벡터 미회수) |
| GS023 | GRAPH | 1 | HAS_REGISTERED_PESTICIDE_USE | – | 37 | 14 | 19 | 79 | max_hops=1 |
| GS026 | GRAPH | 1 | CONTROLS_TARGET | – | 8 | – | 12 | 257 | max_hops=1, 경쟁 심함 |
| GS027 | GRAPH | 2 | AFFECTED_BY_INSECT | – | 69 | – | 69 | 257 | 관계 조합 오류, 경쟁 심함 |
| GS035 | GRAPH | 2 | HAS_REGISTERED_PESTICIDE_USE | – | 35 | 7 | 42 | 81 | 관계 조합 오류 |
| GS036 | HYBRID | 2 | HAS_ENVIRONMENT_REQUIREMENT | – | 120 | – | 9 | 201 | 관계 조합 오류 |
| GS054 | GRAPH | 2 | AFFECTED_BY_INSECT | – | 186 | 27 | 34 | 245 | 관계 조합 오류 |
| GS057 | HYBRID | 2 | CONTROLS_TARGET | **1** | 1 | n/a | n/a | n/a | 현 코드에서 이미 해소 |
| GS059 | GRAPH | 1 | HAS_REGISTERED_PESTICIDE_USE | – | 3 | 12 | 37 | 245 | max_hops=1 |
| GS061 | GRAPH | 2 | AFFECTED_BY_DISEASE | **1** | 1 | 1 | 1 | 200 | 현 코드에서 이미 해소 |
| GS062 | HYBRID | 1 | HAS_REGISTERED_PESTICIDE_USE | – | 8 | 11 | 13 | 46 | max_hops=1 |
| GS066 | HYBRID | 2 | AFFECTED_BY_DISEASE | 6 | 2 | 3 | 27 | 79 | Top-5에서 1칸 밀림 |
| GS067 | HYBRID | 2 | HAS_REGISTERED_PESTICIDE_USE | – | 74 | – | 90 | 117 | 관계 조합 오류, 경쟁 심함 |

## 5. 유형 분류

| 유형 | 문항 | 건수 |
|---|---|---:|
| 2-hop 미탐색(라우터 `max_hops`/관계 타입이 하드 필터로 작동) | GS012, GS018, GS023, GS026, GS027, GS035, GS036, GS054, GS059, GS062, GS067 | 11 |
| 관계 필터로 검색 결과 0건 | GS011 | 1 |
| 라우팅 오류 + 벡터 미회수 | GS020 | 1 |
| Top-5 경계에서 밀림(현 코드 6위) | GS066 | 1 |
| 현 코드에서 이미 Top-1로 해소(2026-11-14 산출물 이후 변경 반영) | GS057, GS061 | 2 |

재측정 기준으로 보면 GS057·GS061은 현재 코드에서 Gold chunk가 1위로 들어온다(2회 반복 동일).
즉 지금 통합 평가를 다시 돌리면 multi-hop 실패는 16건이 아니라 **13~14건**으로 집계된다.
16건은 2026-11-14 산출물 기준 수치다.

## 6. 우선 조치 제안

영향 큰 순서.

1. **`relation_types`를 하드 필터에서 선호 신호로 바꾼다.**
   `graph_retriever.search`의 2-hop 분기에서 라우터가 준 타입으로 0건이면, 현재는 같은 타입으로
   한 번 더 시도한 뒤 1-hop으로 내려간다. broad 재시도를 **타입 필터 없이** 수행하도록 고치면
   설정 B에 해당해 11건이 2-hop 후보를 갖게 된다. 가장 적은 변경으로 가장 큰 효과.
2. **`max_hops`를 라우터 출력에 그대로 의존하지 않는다.** 방제 약제·사용 기준을 묻는 질문
   (`pesticide_target` 분기 조건)은 `max_hops`를 최소 2로 올린다. 6건이 영향권.
3. **GS011 류의 공집합 방지.** 시작 라벨에서 나갈 수 없는 관계 타입만 지정된 경우(`Crop`에
   `CONTROLS_TARGET` 없음) 필터를 버리고 재탐색한다.
4. **형제 경로 순위 경쟁 해소.** 1~3을 적용해도 GS018·GS027·GS067처럼 형제 경로 200개 이상인
   문항은 Gold가 69~135위에 머문다. 질문에 제품 식별 단서가 없어 랭킹만으로는 한계가 있으므로,
   `use_timing`·`use_count`·`dilution` 같은 사용 기준 속성과 질문 조건을 맞추는 재랭킹 신호를
   추가하거나, 평가 기준을 경로 집합 단위로 재검토할 필요가 있다(`docs/multihop-retrieval-report.md`
   5절의 비유일성 이슈와 동일 사안).
5. **GS020 라우팅 교정.** hybrid 전략 문항이 VECTOR로 떨어지는 조건을 라우터 규칙에서 확인한다.

후보 풀 확대(설정 C·D)는 단독으로는 권하지 않는다. (아래 7절 참고) 기존 보고서(`docs/multihop-retrieval-report.md`
3.5절)에서 풀 확대가 통합 성공률을 9/17 → 7/17로 퇴행시킨 기록이 있어, 1~3번의 경로 생성 교정을
먼저 넣고 풀 크기는 그대로 둔 상태에서 재측정하는 순서가 안전하다.

## 7. 적용된 교정 (2026-11-15)

6절의 1·2·3·5번을 반영했다. 4번(형제 경로 순위 경쟁)과 후보 풀 확대는 적용하지 않았다.

### 변경 내용

**`src/retrievers/graph_retriever.py` — `GraphRetriever.search`**

- 질문 의도가 다중 관계(`explicit_multihop` / `variety_region` / `pesticide_target`)면
  `max_hops`를 최소 2로 올린다(`effective_max_hops`). 라우터가 `max_hops=1`을 줘도 2-hop 분기에
  진입한다.
- 2-hop 탐색을 **라우터 관계 타입 → 도메인 기본 조합 → 타입 무제한** 순으로 완화한다.
  기존에는 실패 시 같은 타입으로 한 번 더 시도한 뒤 1-hop으로 내려가, 관계 조합이 틀리면
  2-hop 후보가 아예 생기지 않았다.
- 1-hop 탐색이 공집합이고 관계 타입 필터가 걸려 있으면 필터를 버리고 한 번 더 찾는다
  (`Crop`에 없는 `CONTROLS_TARGET` 단독 지정으로 결과 0건이 되던 GS011 유형).
- 진단에 `path_search_strategy`, `max_hops_used`를 추가했다.

**`src/retrievers/router.py` — `JevQuestionRouter.route_query`**

- 분류기가 VECTOR를 고르고 질문이 "등록된 약제/농약"을 묻는 경우 HYBRID로 올린다
  (`REGISTERED_USE_PATTERN`). 등록 사용 기준은 `Crop→PesticideUse→대상` 관계에만 있다.
  "등록"을 묻지 않는 일반 살포 시기 질문은 VECTOR를 유지한다.

회귀 테스트 5건 추가(`tests/test_graph_retriever.py` 3건, `tests/test_router.py` 2건).
전체 스위트 **476 passed**.

### 측정 결과

라우터는 LLM 기반이라 호출마다 결과가 흔들린다. 검색기 변경만 분리하려고 라우터 판단을
한 번 받아 고정(pin)한 뒤 before/after를 돌렸다. REJECT 제외 57문항, 검색만 실행, `limit=10`.

| 지표 | before | after |
|---|---:|---:|
| chunk hit@5 (전체) | 30/57 | **32/57** |
| chunk hit@10 (전체) | 37/57 | **38/57** |
| multi_hop hit@5 | 5/19 | **6/19** |
| relation hit@5 | 10/19 | **11/19** |
| fact hit@5 | 14/18 | 14/18 |
| 검색 결과 0건 문항 | GS011, GS044 | **없음** |
| 진단 16문항 중 Gold 2-hop 경로가 후보에 들어온 수 | 0/16 | **14/16** |

문항별 변화(pin 기준):

| 문항 | before | after | 비고 |
|---|---:|---:|---|
| GS011 | – | **2** | 결과 0건 → Top-5 진입 |
| GS044 | – | **1** | 결과 0건 → 1위 (relation 문항, 같은 원인) |
| GS015 | 1 | 5 | 여전히 Top-5 (2-hop 경로가 들어와 순위만 이동) |
| GS057 | 1 | 5 | 여전히 Top-5 |
| GS054 | 6 | 9 | before/after 모두 Top-5 밖 |
| GS066 | 6 | – | before/after 모두 Top-5 밖 |

GS020은 라우팅이 VECTOR → HYBRID로 교정됐지만 Gold chunk 순위는 여전히 Top-10 밖이다
(최대 후보 풀에서 92위). 순위 경쟁 문제라 6절 4번 없이는 해결되지 않는다.

### 남은 한계

- 통과 등급이 바뀐 문항은 +2(GS011, GS044)이고, 등급이 내려간 문항은 없다. 다만 GS015·GS057은
  1위 → 5위로 여유가 줄었다.
- 이제 Gold 2-hop 경로는 14/16에서 후보에 들어오지만 순위가 7~135위다. **병목이 후보 회수에서
  랭킹으로 옮겨갔을 뿐이고**, 형제 경로 46~257개 사이에서 Gold를 올릴 신호는 아직 없다.
  6절 4번(사용 기준 속성 기반 재랭킹 또는 경로 집합 단위 채점)이 다음 작업이다.
- 위 수치는 검색 단계만 측정한 값이다. 통합 RAGAS 재실행(`scripts/evaluation/evaluate_integrated_ragas.py`)은
  아직 돌리지 않았다.

## 8. 이름·사용 조건 기반 그래프 랭킹 강화 (2026-11-15)

6절 4번의 첫 단계로 재랭킹 신호를 추가했다. 적용 순서는 요청받은 1→4번 그대로다.

### 8.1 변경 내용 (`src/retrievers/graph_retriever.py`)

**1) 명시적 이름 정확 일치 보너스** — `_name_match_bonus`
- 품목명(`product`), 상표명(`brand`), 유효성분(`active_ingredient_english`), 품종명
  (`source_variety_name`), 방제 대상명(`target_name`), 제조사(`company`)가 **질문에 직접
  등장할 때만** 적용한다.
- 완전 일치 6.0 > 별칭 일치 4.0(`aliases`/`synonyms`/`_variants`) > 부분 일치 2.0.
  작물명 일치는 같은 작물 후보를 구별하지 못하므로 2.0으로 따로 둔다.
- 경로 전체에서 **가장 강한 일치 한 번만** 반영한다(노드가 많은 경로가 유리해지지 않도록).
- 후보를 제거하지 않고 재랭킹 보너스로만 쓴다.

**2) 농약 사용 속성 기반 재랭킹** — 기존 ±1.5 일괄 가산을 필드별 가중치로 교체
- `use_timing` 2.0, `use_count` 2.0, `dilution` 2.0, `application_method` 1.5,
  `registration_no`·`product`·`brand` 1.0. 없으면 가중치의 0.75배를 감산.
- 질문이 요구한 필드를 **모두** 갖춘 기록에 완전성 가산 +2.0.

**3) 질문에 적힌 조건값 일치** — `_requested_condition_values` / `_condition_value_score`
- 질문에서 희석배수(`1000배`), 사용 횟수(`3회`), 살포 간격(`10일 간격`),
  수확 전 일수(`수확 7일 전`), 시기 표현(`월동 후`·`발생 초기` 등)만 읽는다.
- 값이 맞으면 +3.0, 틀리면 −1.0. 시기 표현은 표기가 다양해 가산만 한다.
- **Gold 값은 쓰지 않는다.** 질문 텍스트에서 파싱한 값만 비교하므로 평가 누수가 없다.

**부수 수정:** 커버리지 속성 목록에서 `classification_paths`·`cultivation_type`을 제외하고
`품종`·`제품`·`품목` 같은 분류어를 불용어에 추가했다. 이 일반 단어들이 일부 후보의 분류 문자열에만
들어 있어, 정작 Gold 품종 노드가 1.2점 뒤처지는 역전이 있었다(GS055가 1위 → 10위로 밀렸고,
수정 후 1위 복귀).

**4) 경로 집합 단위 평가 병행** — `scripts/evaluation/diagnose_multihop_top5_misses.py`
- `_alternative_path_rank`로 "같은 작물 → **다른** 등록 약제 → 같은 방제 대상" 경로의 순위와
  `same_target_rank`를 함께 기록한다. 단일 Gold exact-match와 **합산하지 않고 따로** 본다.

회귀 테스트 7건 추가. 전체 스위트 **483 passed**.

### 8.2 측정 결과

라우터 판단을 고정한 동일 조건, REJECT 제외 57문항, 검색만 실행, `limit=10`.

| 지표 | 원본 | 7절 적용 | 8절 적용 |
|---|---:|---:|---:|
| chunk hit@5 (전체) | 30/57 | 32/57 | **33/57** |
| chunk hit@10 (전체) | 37/57 | 38/57 | **40/57** |
| multi_hop hit@5 | 5/19 | 6/19 | **7/19** |
| multi_hop hit@10 | 7/19 | 7/19 | **8/19** |
| relation hit@5 | 10/19 | 11/19 | 11/19 |
| fact hit@10 | 16/18 | 17/18 | 17/18 |

7절 대비 변화: GS054 9위 → **1위**, GS059 미회수 → **8위**, GS021 미회수 → **6위**.
순위가 내려간 문항은 없다. 3회 반복에서 33/57·40/57로 동일했으나, GS054·GS059는 동점 경계에
있어 한 번은 32/57로 측정된 적이 있다.

### 8.3 이름 일치 보너스의 실제 효력

질문에 품목명을 넣어 확인한 결과(평가 지표가 아니라 기능 확인용):

| 문항 | 원 질문 | + 품목명 명시 |
|---|---:|---:|
| GS026 | 미회수 | **1위** |
| GS012 | 미회수 | **3위** |
| GS067 | 미회수 | **2위** |

**상표명만 명시하면 효과가 없다.** 후보 사전 정렬 Cypher가 노드 **이름**만 보고 상위 90개를
자르는데, 상표명은 노드 이름이 아니라 속성에만 있어 Gold 경로가 후보 풀에 들어오지 못한다.
사전 정렬에 속성 일치를 0.25 가중치로 더하는 실험을 했으나, `symptoms`·`ecology` 같은 장문
속성에서 오탐이 생겨 GS067이 2위 → 미회수로 퇴행했다. **반영하지 않고 롤백했다.**
상표명 검색이 필요하면 Cypher 문자열 비교 대신 식별자 전용 속성이나 전문 검색 인덱스를
노드에 따로 만드는 편이 맞다.

### 8.4 공식 등록 상태로는 Gold를 가릴 수 없다 (측정 근거)

제품 간 우선순위를 공식 등록 상태로 정할 수 있는지 확인했다. 16문항의 Gold 제품과 형제 경로
전체(4~257개)를 조회한 결과:

- `recommendation_eligible`: Gold·형제 **전부 `false`**
- `registration_status`: Gold·형제 **전부 동일 값**
- `audit_disposition`: Gold·형제 **전부 `null`**

즉 등록 상태·추천 적격 여부에는 **식별력이 전혀 없다.** 질문이 제품을 지목하지 않는 한,
수십~수백 개 형제 경로 중 Gold 하나를 올릴 근거가 데이터에 없다.

### 8.5 경로 집합 단위 지표 (6절 4번)

파이프라인 Top-10 기준, 16문항에 대해 두 지표를 따로 측정했다.

| 지표 | 값 |
|---|---:|
| 단일 Gold exact-match Top-5 | 4/16 (GS011, GS054, GS057, GS061) |
| "같은 작물 → 다른 등록 약제 → 같은 방제 대상" 유효 대체 경로 **1위** | 14/16 |

대체 경로 14/16은 **정답률이 아니다.** 질문이 제품을 구별하지 못할 때 검색이 구조적으로
타당한 경로를 최상위에 올리고 있다는 뜻일 뿐이며, 그 약제가 실제로 적합한지·현재 등록이
유효한지는 8.4에서 보듯 이 데이터로 입증되지 않는다. 공식 성능은 기존 exact-match 기준을
유지한다.

### 8.6 다음 단계

1. 평가 기준 결정이 먼저다. 제품 식별 단서가 없는 문항을 단일 Gold exact-match로 채점할지,
   경로 집합(공식 등록 약제 집합) 단위로 채점할지 팀에서 정해야 한다. 8.4 때문에 랭킹만으로는
   더 올라가지 않는다.
2. 평가셋을 고치는 쪽이라면, 해당 문항에 제품 식별 단서(품목명·상표명·유효성분)를 넣는 것으로
   충분하다. 8.3에서 품목명 명시 시 1~3위로 올라오는 것을 확인했다.
3. 상표명·유효성분 검색을 지원하려면 후보 사전 정렬 단계를 고쳐야 한다(8.3).
4. 통합 RAGAS 재실행은 아직 하지 않았다. 7·8절 수치는 모두 검색 단계만 측정한 값이다.
