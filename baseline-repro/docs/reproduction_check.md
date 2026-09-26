# 재현 점검: 우리 결과 vs 각 논문 보고 수치 (2026-09-26)

**한 줄 요약:** 논문 수치와 "허용 오차 안에서" 맞는 것은 **Optimus-1 (자기 환경 Env O + 저자 공개 full memory)의 Overall 하나뿐**이다. 그마저도 그룹별 모양은 다르다(쉬운 그룹은 논문보다 유의하게 낮고, 어려운 그룹은 같거나 높음). **MineEvolve, JARVIS-1, Optimus-3은 재현되지 않았다**(시드·디버깅 차이로 설명할 수 있는 범위를 크게 벗어남). DEPS는 Overall만 신뢰구간 안에 걸친다. 그리고 **모든 방법에서 Wooden/Stone이 논문보다 일관되게 낮다** → 방법 개별 문제가 아니라 공통 요인(환경·컨트롤러·판정 기준)이 섞여 있다.

재생성: `python analysis/compare_papers.py` (입력 `analysis/our_numbers.json`, `analysis/paper_numbers.json`; 상세 CSV `analysis/out/compare_groups.csv`, `compare_tasks.csv`).

---

## 1. 비교 방법

- **우리 수치:** 태스크당 시드 1개. 메모리 누적 방법(MineEvolve, Optimus-1)은 태스크 순서 3개 → 태스크당 3회, 나머지(DEPS, JARVIS-1, Optimus-3)는 1회. 성공 = 목표 아이템이 인벤토리에 들어온 순간 (D12, 모든 방법 동일).
- **허용 오차 판단:** 그룹 성공률에 **Wilson 95% 신뢰구간**을 붙이고, 논문 값이 구간 **밖**이면 "시드·잡음으로 설명 안 되는 차이"로 본다. 태스크 단위는 논문이 태스크별 수치를 준 경우만 **정확 이항검정(p<0.05면 불일치)**.
- **주의 — 검정력:** 그룹당 태스크 6–16개 × 1–3회라 구간이 넓다. 특히 성공률 0% 근처 그룹(Gold/Redstone/Diamond)은 0%여도 상한이 16–39%라 "OK"가 나오기 쉽다 → **어려운 그룹의 "구간 안"은 '일치'가 아니라 '구분 불가'로 읽어야 한다.** JARVIS-1 태스크 검정(태스크당 1회)은 사실상 아무것도 기각할 수 없어 의미 없음.

## 2. Overall (70개 태스크, task-count 가중)

| 방법 | Env | 우리 (95% CI) | 비교 논문 수치 | 판정 |
|---|---|---|---|---|
| **Optimus-1** (full memory) | O | **46.2** (39.6–52.9) | 46.7 (MineEvolve T4, Gemini-3-Flash) | ✅ 구간 안 |
| Optimus-1 (full memory) | M | 30.5 (24.6–37.0) | 46.7 | ❌ 낮음 |
| Optimus-1 (full memory + goalfix) | O / M | 43.8 / 28.1 | 46.7 | ✅ / ❌ |
| Optimus-1 (empty memory, 원본/goalfix) | O / M | 5–12 | (논문은 pre-built memory 사용) | 비교 대상 아님 |
| **MineEvolve** | M / O | **11.9 (8.2–17.0) / 10.0 (6.6–14.8)** | **52.0** | ❌❌ 크게 낮음 |
| **DEPS** (포팅) | M / O | 25.7 (16.9–37.0) / 18.6 (11.2–29.2) | 29.7 | △ 구간 안(경계) |
| **JARVIS-1** (포팅) | M / O | 14.3 (7.9–24.3) / 10.0 (4.9–19.2) | 42.0 | ❌ 크게 낮음 |
| **Optimus-3** (자체 모델, 별도 설정) | C3 | 28.6 (19.3–40.1) | ≈51 (Optimus-3 T.III 그룹값을 우리 70개 비율로 가중) | ❌ 낮음 |

참고: MineEvolve 논문(Table 4)의 baseline 수치는 각 원 논문 값과 거의 같다(예: Optimus-1 행 ≈ Optimus-1 논문 Table 1, JARVIS-1 Stone 88.58 vs 원논문 88.69). 그래서 "MineEvolve Table 4 대비"는 사실상 "원 논문 대비"와 같다. Optimus-1 논문의 "Overall"은 Iron/Gold/Diamond/Redstone/Armor 5그룹 평균이라 정의가 다르다(Optimus-1 22.26).

## 3. 그룹별 (우리 [95% CI] vs 논문, ↓/↑ = 구간 밖, OK = 구간 안)

| 우리 / 비교 논문 | Wood | Stone | Iron | Gold | Redst. | Diam. | Armor |
|---|---|---|---|---|---|---|---|
| **O/Optimus-1 full** vs Optimus-1 논문 | 67 [50–80] vs 99 ↓ | 77 [59–88] vs 92 ↓ | 56 [42–69] vs 47 OK | 14 [5–35] vs 9 OK | 17 [6–39] vs 25 OK | **33 [17–55] vs 12 ↑** | 31 [19–46] vs 19 OK |
| M/Optimus-1 full vs Optimus-1 논문 | 76 vs 99 ↓ | 47 vs 92 ↓ | 44 vs 47 OK | 0 vs 9 OK* | 0 vs 25 ↓ | 0 vs 12 OK* | 10 vs 19 OK |
| **M/MineEvolve** vs MineEvolve 논문 | 61 [44–75] vs 99 ↓ | 13 [5–30] vs 93 ↓ | 2 [0–11] vs 55 ↓ | 0 vs 14 OK* | 0 vs 34 ↓ | 0 vs 13 OK* | 0 [0–9] vs 27 ↓ |
| O/MineEvolve vs MineEvolve 논문 | 33 vs 99 ↓ | 20 vs 93 ↓ | 6 vs 55 ↓ | 0 vs 14 OK* | 0 vs 34 ↓ | 0 vs 13 OK* | 3 vs 27 ↓ |
| M/DEPS vs MineEvolve T4 | 46 vs 84 ↓ | 20 vs 70 ↓ | **44 vs 18 ↑** | 0 vs 4 OK* | 0 vs 10 OK* | 0 vs 3 OK* | **31 vs 5 ↑** |
| O/DEPS vs MineEvolve T4 | 27 vs 84 ↓ | 40 vs 70 ↓ | 25 vs 18 OK | 0 vs 4 OK* | 0 vs 10 OK* | 0 vs 3 OK* | 15 vs 5 OK |
| M/JARVIS-1 vs JARVIS-1 논문 | 73 vs 89 OK | 20 vs 89 ↓ | 0 vs 35 ↓ | 0 vs 7 OK* | 0 vs 18 OK* | 0 vs 9 OK* | 0 vs 13 OK* |
| O/JARVIS-1 vs JARVIS-1 논문 | 64 vs 89 ↓ | 0 vs 89 ↓ | 0 vs 35 ↓ | 0 vs 7 OK* | 0 vs 18 OK* | 0 vs 9 OK* | 0 vs 13 OK* |
| C3/Optimus-3 vs Optimus-3 논문 | 64 vs 99 ↓ | 60 vs 95 ↓ | 25 vs 55 ↓ | 0 vs 10 OK* | 0 vs 29 OK* | 14 vs 15 OK | 15 vs 23 OK |

\* 0%라서 구간 상한이 넓어 생긴 "OK" — 일치가 아니라 **검정력 부족**.

## 4. 태스크별 (논문이 태스크별 수치를 준 경우만)

- **Optimus-1** (논문 부록 F.1, GPT-4V, 태스크당 30–40회; Wood 9개·Stone 8개만 추출 가능, 나머지 그룹 표는 가져오지 못함):
  - O / full memory: **14개 중 11개**가 논문 SR과 모순 없음(이항검정 p≥0.05). 불일치 3개는 모두 우리가 낮은 쪽.
  - M / full memory: 14개 중 9개. empty memory: 0–4개.
- **JARVIS-1** (논문 부록, Stone·Iron 전부 + Wood/Gold 일부 = 33개): 33개 중 32개 "모순 없음"이지만 **태스크당 1회라 검정력이 사실상 0** → 판단 근거로 쓰면 안 됨. 그룹 수준(Stone 0–20 vs 89, Iron 0 vs 35)이 실질적 판단.
- **MineEvolve 논문에는 태스크별 수치가 없다**(그룹 수준만). **DEPS**는 태스크 체계가 다르고(MT1–MT8, MineDojo 1.11) 파이프라인 전체의 태스크별 수치가 없어 태스크 비교 불가. **Optimus-3**은 우리 70개와 대응하는 태스크별 표가 없다(Table IV는 별도 5개 과제).

## 5. 방법별 판정

| 방법 | 판정 | 근거 |
|---|---|---|
| **Optimus-1** (Env O, full memory) | **부분 재현** | Overall 일치(46.2 vs 46.7), Iron/Gold/Redstone/Armor 일치, Wood/Stone 유의하게 낮고 Diamond는 오히려 높음. 태스크별 11/14 일치. → 재현 수치로 쓸 만하나 "분포가 다르다"는 각주 필요 |
| Optimus-1 (Env M) | 미재현 | Env M의 STEVE-1 설정(cond_scale 4.0)이 전 방법을 불리하게 만듦 |
| **MineEvolve** | **미재현 (큰 차이)** | 판정 가능한 5개 그룹 전부 유의하게 낮음. Stone 13–20 vs 93. 공개 코드가 논문 결과를 낸 코드가 아닐 가능성이 높음(크래프트 stub, 성공 판정 버그, 동작 안 하는 auto-pickaxe가 공개본에 있었음) |
| DEPS (포팅) | 애매 | Overall은 구간 안이지만 쉬운 그룹은 낮고 Iron/Armor는 높음 — 우연의 상쇄. 원 코드가 MineDojo/자체 컨트롤러라 포팅 자체가 다른 시스템 |
| JARVIS-1 (포팅) | 미재현 | Stone/Iron이 크게 낮음. 공개본에 플래너가 없어 논문 부록 프롬프트로 재구성한 포팅(쿼리 생성·시각 기술자 미공개)이라 원 시스템과 차이가 큼 |
| Optimus-3 (자체 모델) | 미재현 | Wood/Stone/Iron 유의하게 낮음. 논문의 67개 벤치·환경과 다르고, 플래너 입력 형식 변환(C1) 필요했음 |

## 6. 공통 요인 (다음 방향 정할 때 중요)

1. **쉬운 그룹이 전부 낮다.** Wood/Stone에서 논문 90%대 vs 우리 13–77%. 모든 방법·두 환경에서 같은 방향 → 방법이 아니라 우리 설정 쪽 요인이 크다:
   - STEVE-1 속도/효율: Env M의 cond_scale 4.0은 6.0 대비 통나무 수확이 절반(통제 실험, D32). Env O(6.0)에서도 여전히 낮음.
   - 짧은 horizon: Wood 2분, Stone 3분(MineEvolve 설정). 논문들의 평균 성공 스텝(예: Optimus-1 나무 곡괭이 ≈1,150 steps)을 보면 꼬리 분포에서 잘린다.
   - 판정 기준이 엄격: "oak log" 과제에서 dark oak log는 실패(Wood 실패 9건 중 3건).
   - 시드 1개(태스크당 월드 1개): 사막·동굴 스폰 같은 불리한 월드가 그대로 반영됨. 논문들은 태스크당 30회 이상.
2. **어려운 그룹은 논문과 같거나 높기도 하다**(Optimus-1 Diamond 33 vs 12, DEPS Iron/Armor). Env O의 광석 생성 규칙(y≤14면 항상 다이아몬드)과 긴 horizon(30분) 영향으로 보임.
3. 즉, "재현 실패"의 상당 부분은 **평가 환경 차이**로 설명될 수 있고, MineEvolve·JARVIS-1은 그 위에 **코드 자체의 차이**가 겹쳐 있다.

## 7. 제안 (결정은 사용자)

- 우리 방법의 비교 대상 수치는 **논문 인용값이 아니라 동일 조건에서 우리가 잰 값**을 쓰는 게 안전. 인용할 수 있는 건 Optimus-1(Env O) Overall 정도.
- 쉬운 그룹 차이의 원인을 가르려면(저비용 순): ① "oak log" 판정 완화 재계산(재실행 불필요), ② Wood/Stone만 태스크당 시드 5개로 재측정(수 시간, API 저렴), ③ Env M을 cond_scale 6.0으로 돌리는 민감도 실험(MineEvolve만).
- MineEvolve 공개 코드 ≠ 논문 코드 가능성은 저자 문의가 가장 빠름.

## 8. 추가 점검 (API·GPU 없이, 09-26)

- **"oak log" 판정 완화 효과는 작다.** wooden_08에서 다른 종류의 통나무를 가진 실패는 방법별 0–3건(총 3개 순서 기준). Wood 그룹(11개 태스크)에서 최대 +1 태스크 수준 → 논문과의 격차(Wood 20–60%p)를 설명하지 못함.
- **horizon이 성공을 자르고 있지 않다.** Wood+Stone 성공은 horizon의 중앙값 10–39% 시점에 일어나고, 마지막 25% 구간 성공은 거의 없음(방법별 0–11건). horizon까지 간 실패는 대부분 **진전 없이 막힌 에피소드**(컨트롤러/상황 문제)라, horizon을 늘려도 크게 안 오를 가능성이 높다.
- 따라서 쉬운 그룹 격차의 유력 후보는 **STEVE-1 설정·환경(월드 시드, 스폰 상황)** 쪽이다. 이를 가르는 API 없는 실험(STEVE-1 단독, Wood+Stone × 시드 3개, Env M 4.0 / Env M 6.0 / Env O; `configs/run_plan_diagD1.yaml`)은 준비돼 있으나, 09-26 06시에 다른 사용자가 GPU 0,1,2,3,6,7을 점유해 즉시 중단했다. GPU가 비면 실행 가능.
