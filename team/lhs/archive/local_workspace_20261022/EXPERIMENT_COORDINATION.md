# Codex · Claude 실험 공유 기록

두 에이전트가 중복 학습·추론과 GPU 충돌을 피하기 위해 함께 갱신하는 파일이다. 시간은 KST로 기록한다.

## 작업 시작 전 규칙

1. 새 실험을 준비하거나 서버 작업을 시작하기 전에 이 파일을 읽는다.
2. 아래 담당 범위와 실행 중인 작업을 확인한다. 같은 데이터 분할·초기 가중치·입력·학습 방법·결합 규칙을 사용하는 작업은 실행 전에 중복 여부를 확인한다.
3. 새 작업의 담당자, 실험 ID, 목적, 데이터, 서버 경로, GPU, 상태를 **실행 전에** 기록한다. 준비 중인 작업도 먼저 등록한다.
4. 서버의 실제 프로세스와 GPU 점유를 확인한다. 이 파일의 상태·PID·잠금 파일만으로 실행 여부를 판단하거나 재실행하지 않는다.
5. 시작, 단계 전환, 실패, 완료, 채택 여부가 바뀌면 자기 항목을 갱신한다. 다른 에이전트의 항목을 추측으로 완료 처리하지 않는다.
6. 다른 에이전트가 갱신했을 수 있으므로 수정 직전에 다시 읽고 자기 항목만 수정한다. 전체 파일을 오래된 사본으로 덮어쓰지 않는다.

## 담당 범위

| 담당자 | 담당 범위 | 중복 방지 |
|---|---|---|
| Claude | 정답 90장 시편 단위 4-fold × 시드 학습, OOF90 생성, OOF 기반 구성·밝기 배정·결합·후처리 선택 설계 및 구현 | 사용자 지정 담당. Codex는 이 설계와 학습을 별도로 구현·실행하지 않는다. |
| Codex | OOF90 외의 성능 향상 실험. 현재 train70 내부 보류 fold에서 HRNet 질감 입력 대조 실험 | Claude는 같은 HRNet 질감 실험을 새로 실행하기 전에 아래 상태를 확인한다. |

## Claude 상태 — Claude가 직접 갱신

- **2번 B4ws 시드 확인 결과 (2026-10-12 11:57 KST):** lite19a valid20 최종 기준이다.
  - 원래 B4 대비 B4ws 효과: 시드 819349+1000×{4,5,6}으로 다시 학습한 3벌이 +0.031±0.002(3/3 양수), 공개판 +0.030이다. 다시 학습한 B4ws는 공개판 대비 −0.001~+0.003이다. B4ws는 유지한다.
  - 안전장치 OOF 재선택(train70만 사용): AB는 OOF 80.072(+0.26), 바깥 추정 80.017(+0.20), 최악 OOF 이미지 −0.86이다. A는 80.129(+0.31), 바깥 추정 80.032(+0.22), 최악 −1.10이다. B는 선택 중이고, 셋 모두 끝나면 valid를 채점한다.

- **R HRNet336 자리를 E HRNet392(공유)로 교체 (anacnu CPU, `~/komap/eseed/r_share.py`): 기각.** lite19a valid20 최종 기준이다.
  - HR392 4 fold로 교체 −0.163, 347까지 제거하면 −0.215, HR392 8 fold로 교체 −0.167이다.
  - 참고: 336과 347을 모두 빼면 −0.453이다. R의 결합 비율이 336에 맞춰져 있어 교체하면 손실이 난다(B4ws를 R에 넣었을 때와 같다).

- **E HRNet392를 R HRNet336으로 공유 (2026-10-12 09:xx KST, anacnu CPU, `~/komap/eseed/e_study_share.py`): 기각.**
  - lite19a valid20 최종 기준 결과: E=0.5 B4ws+0.5 R336은 −0.179(최악 −1.32), +0.25 R336+0.25 R347은 −0.168, seed2 자리만 336으로 바꾸면 −0.048, HR 모두 제거는 −0.418이다.
  - E의 HRNet392 fold 4개는 유지한다.

- **5번 결과 (2026-10-12 08:23 KST): OOF 재선택 기각.**
  - fold 4개 학습 약 4시간, OOF 예측 15분, 선택 2시간 10분이 걸렸다.
  - OOF 1단계: 공개 설정 79.817, OOF 선택 80.189(+0.37), 바깥 fold 추정 80.072(+0.25). 바뀐 설정은 al3ni 0.7→0, al 0.6→1, eut 0.7→0.25, gamma329/336/347, prior 1.25→1.0, al_delta, eut_delta, eut_edge_blend다.
  - valid20(공개 가중치, E+F): 공개 설정 80.5236, OOF 선택 79.8293(−0.694). 13/20장은 좋아졌지만 A71 −13.0, A11_step1 −5.5다. 두 장을 빼면 나머지 18장 평균은 약 +0.26이다. #44와 같은 양상이다(A71·A11).
  - 1번: HR seed1 제거 −0.055, seed2 제거 −0.081, HR 모두 제거 −0.418, B4 제거 −0.139. 모두 유지한다.

- **anacnu 자동 체인 (2026-10-12 02:5x KST 등록, 실행 중):** 순서는 5 → 1 → 2 → 3이다. 앞 단계가 끝나면 사람 개입 없이 이어진다.
  - **5 (`~/komap/oof`):** lite19a R 결합 비율을 OOF로 다시 고른다.
    - `oof_design`에 `--fold-manifest`(model303 fold)를 추가했다. 이 분할로 B4 이외 R 구성원 15단계를 fold 4개에 대해 seed 0으로 학습한다(B4는 공개 fold 모델을 재사용, fold당 약 1시간).
    - 이어서 OOF를 예측하고, 1단계 목표(R + E OOF 덮어쓰기)로 좌표 상승 선택과 바깥 fold 추정을 한다. 마지막으로 valid20을 1회 채점한다(`select_l19a.json`, `valid.log`).
  - **1 (`~/komap/eseed/item1.txt`):** E에서 HRNet392 seed1, seed2, 둘 다를 빼거나 B4를 빼고 채점한다(저장 확률).
  - **2 (`item2.txt`):** B4ws를 시드 819349+1000×{4,5,6}으로 fold 4개씩 다시 학습한다. 원래 B4 대비 B4ws 효과의 시드 분포를 본다.
  - **3 (`item3.txt`):** E가 쓰는 HRNet392 fold 4개(s1 f2·f4, s2 f1·f4)를 시드 세트 2개(S=5,6)로 처음부터 다시 학습한다. HRNet392 기여의 시드 분포를 본다.
  - 종합 결과는 `~/komap/eseed/summary.txt`에 모인다.

- **F 시드 확인과 E B4ws fold 부분집합 (2026-10-12 01:4x KST, anacnu `~/komap/fseed`, `~/komap/add/eval_e_b4.*`):**
  - F: 원래 시드 81349로 다시 학습하면 5080에서 SHA는 다르지만(c2a8d9…) 점수는 같다(80.5237). F 기여(lite19a, F 없음 80.4966%)는 시드 4031~4035에서 +0.026±0.001(5/5 양수), 공개판 +0.027이다. F는 유지한다.
  - E B4ws fold 부분집합 14가지: fold 1개만 쓰면 −0.005~−0.091이다. 2개는 13 +0.009, 14 +0.014, 나머지는 음수다. 3개는 −0.029~+0.006이다. valid로 fold를 고르는 것이라 채택하지 않고 4개 평균을 유지한다.

- **model269 시드 확인 (2026-10-12 01:3x KST, anacnu `~/komap/m269`):**
  - 공개판 269의 부모 257은 팀원 서버(route_v2 실행)에만 있어서, P100의 257(09-23 계보)에서 시드 4021~4025로 재학습했다. 레시피는 그대로다.
  - lite19a 기준 269 기여(no269 = eutectic_gamma 0, 80.4369%): 새 시드 +0.048±0.005(5/5 양수), P100 기존 269 +0.065, 공개판 +0.087.
  - 결론: 269는 유지한다. 단독 점수는 새 시드(77.1~77.3)가 공개판과 비슷하거나 높은데도 결합 기여는 공개판이 가장 크다. 336과 같이 결합 설정이 공개판에 맞춰진 양상이다.

- **lite19a + 추가 모델 조합 탐색 (2026-10-12 00:5x KST, anacnu conda `komap`, `~/komap/add`):**
  - lite19a를 원본부터 다시 추론한 대조군: valid 80.5236%(R 80.1965%).
  - 공개 가중치 후보 15개를 하나씩 넣거나 두 개씩 넣은 121가지를 채점했다(E+F 포함). 후보는 R B4 미사용 fold 6, 고해상도 restart51·model109·seed456 평균, 336 시드 9개 평균, E HR392 미사용 fold 4, E 원래 B4-329다.
  - 결과: 단독 최고는 model109 +0.005(최악 −0.31), 쌍 최고는 model109+HR s1f3 +0.0098이다. 사전 기준(+0.01, 최악 −0.1)을 통과한 것은 없어 선택 없음. LOSO 재선택은 −0.012다.
  - 336 시드 평균을 넣으면 −0.137이다(결합 설정이 공개 336에 맞춰져 있음). R B4 fold 추가는 −0.003~−0.072다.
  - test 후보 확률 계산은 선택된 조합이 없어 중단했다.

- **anacnu(RTX 5080 1장, `ssh anacnu`, `~/komap/hr336`) 시드 연구 (2026-10-11 20:5x KST 시작, 실행 중):** 공개 lite20 가중치 기준. P100 GPU는 쓰지 않는다.
  1. 공개 lite20 R(구성 요소 확률 보존)과 E를 추론한다. R 단독 valid 80.2109%로 P100 80.2107%와 같다.
  2. 팀원 HRNet336 시드 3407~3414를 336 자리에 끼우고, 시드 앙상블(8개, 9개)을 만든다. 각각 347 있음/없음으로 E+F 결합해 채점한다(`fuse_variants.py`).
  3. HRNet347을 시드 4001~4005로, restart61(pairwise)을 시드 4011~4015로 공개 레시피 그대로 재학습한다. 시드마다 기여(lite20 − no347, lite20 − lite19p)를 잰다(`train_extra.sh`, `eval_extra.sh`, `fuse_extra.py`).
  - 시드별 결과로 구성을 고르지 않고, 기여의 평균과 흔들림만 본다.
  - **결과 (22:4x KST 완료):** 5080 fresh lite20 80.5302%(P100 80.5301%과 같음).
    - 336 시드 8개 평균 80.227±0.115. 팀원 기록과 0.05%p 안에서 일치한다. 공개판은 +0.30%p 높다. 시드 앙상블은 8개 80.367, 9개 80.381.
    - 347 기여: 336 시드 8개에서 +0.029~+0.055%p. 347을 다시 학습한 시드 5개에서는 +0.0257±0.0020(5/5 양수, 공개판 +0.024).
    - pairwise(restart61) 기여: 다시 학습한 시드 5개에서 +0.0074±0.0009(5/5 양수, 공개판 +0.0097). valid에서는 A71 한 장에만 작용하고, test에서는 A51·A167에 작용한다.
    - 구간 규칙 없는 고해상도 경로 (22:5x): A(모든 이미지에 restart61)는 밴드판 대비 시드 5개 평균 −0.0023±0.0070, 공개판 −0.0065. B(51·61 평균)는 −0.0050±0.0041. 둘 다 사전 기준(−0.02 이내)을 통과했다. A는 restart51이 필요 없어 가중치가 19개다.
    - 결과는 로컬 scratchpad `anacnu_results/`로 백업했다. 사용자 지시로 anacnu의 `~/komap`(venv·데이터·가중치·결과), `~/.cache/pip`, `~/.nv`를 삭제해 원상복구했다(SSH 키는 유지). 이후 anacnu는 conda + docker 환경으로만 실행한다.

- **lite18 계산 (2026-10-11 19:xx, CPU만, 저장 확률):** lite20에서 R의 pairwise(밝기 159~161 구간)와 HRNet347을 뺀 구성. 서버 `claude_prune_seed456_m109_20261011/scan5`, `fuse5.py`, `lite18_result.json`.
  - 대조군 offline lite20 80.5304%(fresh 80.5301%와 일치). lite18 **80.4974%**, 차이 −0.033%p (시편 bootstrap 95% [−0.054, −0.013]), 개선 7 / 악화 13, 최악 A170 −0.138%p. R 단독 차이는 −0.025%p.
  - 사전 기준(평균 손실 0.05%p 이내)은 통과했다. 채택 여부는 사용자가 결정한다.
  - test10 (GPU 모두 사용 중이라 `test_compare_20261008/probe` 저장 확률로 재조합, 같은 방식의 lite20 R이 fresh와 30픽셀 차이): 최종 마스크가 lite20 대비 7,175 / 9,074,964픽셀(0.079%) 다름. A51(pairwise 사용 이미지) 3,202픽셀, 나머지 9장 258~796픽셀. 결과 `claude_prune_seed456_m109_20261011/test18/`.

- **lite20_r70 재현 ZIP (2026-10-11 18:5x, 미실행·GPU 사용 없음):** R의 B4 fold 선택 편향을 없애기 위해 322·326·329를 계열당 train70 전체 모델 1개로 바꾼 변형(가중치 20→17)이다. 로컬 `Model349_lite20_r70_bundle_code.zip`(SHA d6599556…)이며 lhs `34c4660`에 업로드했다.
  - `run_r70.sh`는 B4 3개를 학습(fold 레시피와 같고 데이터는 train70, 반복은 16,000×70/52.5 ≈ 21,000회, valid 미사용)한 뒤 같은 나머지 가중치로 lite20_r70과 lite20을 둘 다 추론해 짝 비교한다.
  - `infer_lite.py`에 선택 설정 `inference.b4_full70`을 추가했다(없으면 기존과 같다). 로컬에서 가짜 GPU·학습으로 흐름만 확인했다.
  - 시드 탐색은 팀원 담당이므로 실행은 팀원에게 맡긴다.

- 마지막 실제 확인: **2026-10-10 11:55 KST** (Claude GPU 사용 없음)
- OOF 설계 코드: **완료, 사용자 지시로 팀원에게 인계.** `Model349_exh14/oof_design/`, `Model349_fusion805/oof_expert/`를 CNUKoMaP/lhs 커밋 `aa418cb`로 업로드했다. 기본은 A 방식(train70 fold × 시드 학습, OOF70 선택, valid20 1회 채점)이고 B 방식(90장)은 옵션이다.
  - smoke(`/workspace/model349_compress/oof_smoke/`, GPU7): prepare와 학습 12/19단계(model6~model242)까지 통과한 뒤 사용자 지시로 18:4x KST에 중지했다. 이후 단계는 팀원이 이어서 검증한다. Claude는 이 학습을 더 실행하지 않는다.
- 현재 작업: 보고서용 lite23+B4ws test10 추론(GPU4, 사용자 승인; 다른 사용자가 메모리 2.5GB만 점유한 유휴 GPU). Al₃Ni 가중 실험(`claude_b4ws_al3ni_cw_20261010`)은 보고서 작업을 위해 대기 중단(미실행). lhs `e570396`: 재학습 키트에 B4-329 R·E 공유(fold·레시피 동일 확인)와 GPU 병렬 묶음 실행 추가.

### Claude 진행·예정

- **`claude_b4ws_al3ni_cw_20261010`** / 상태: **실행 전 등록 (18:1x KST). 빈 GPU 대기**
  - 가설: 최종 결합에서 E는 Al₃Ni·Al 픽셀만 덮어쓴다. B4 추가 학습(#32)에서 CE의 Al₃Ni 가중치를 2배로 주면 E의 Al₃Ni 판단이 좋아질 수 있다.
  - 레시피: #32와 동일(부모 model329 fold k, train_fold k, 2,000회, lr 3e-5, BN 고정, seed 819349). 차이는 `--ce_weights 1,2,1,1`(라벨 순서 Al, Al₃Ni, PSi, Eut)뿐이다. 변형은 이것 하나로 고정한다.
  - 코드: `train_bnfreeze_cw.py`(CE weight 인자만 추가). 출력 `/workspace/komap_model1/outputs/claude_b4_329_warmstart_bnfreeze_cw/fold{1..4}`
  - GPU: 0·3(Codex)을 제외한 4·6·7 중 메모리 100MiB 미만, 사용률 0%가 3회 연속 확인된 GPU 하나에서만 시작한다. 다른 사용자 GPU는 쓰지 않는다.
  - 평가: OOF70에서 #32 대비 평균과 클래스별 IoU를 비교한다. 다음 단계 조건은 Al₃Ni IoU +0.3%p 이상이면서 평균이 −0.05%p 이내일 때다. 조건을 만족하면 lite23 고정 후보 1개를 valid20으로 채점한다.

- **`claude_lite23_eratio_20261010`** / 상태: **완료 — 기각 (18:0x KST)**. w=0.4 80.5479, 0.5 80.5464, 0.6 80.5215, 0.7 80.4653. nested −0.014 [−0.033, +0.002]. 0.5 유지. `claude_lite23_eratio_20261010/result.json`.
  - 가설: B4가 추가 학습으로 강해졌으므로(OOF +0.38), E 안의 B4 비율 0.5가 최적이 아닐 수 있다.
  - 후보(미리 고정): B4 비중 w ∈ {0.4, 0.5, 0.6, 0.7}, HR seed1·seed2는 나머지를 반씩. 기준은 lite23 + B4ws(w=0.5, 80.5464%). F·규칙·R은 고정.
  - 방법: lite23 fresh E 확률에서 HR 부분 = E − 0.5·B4(원래)를 분리하고, w·B4ws + (1−w)/0.5·HR로 재구성. 저장 확률만, 새 추론 없음.
  - 평가: valid20 전체 + nested LOSO(시편을 빼고 4개 중 재선택). 채택 기준: nested가 w=0.5 대비 +0.02%p 이상이고 하한 0 이상.
  - Codex의 E 평균 방식(powerpool, 비율 고정) 실험과 다르다. 비율만 바꾼다.

- **`claude_lite23_b4full70_20261010`** / 상태: **취소 — 사용자 지시로 팀원이 실행 (16:5x KST).** 대기 프로세스를 종료했고 학습은 시작하지 않았다. 스크립트와 판단 기준은 lite23_b4ws ZIP의 `full70_experiment/`로 넘긴다. Claude·Codex는 이 실험을 중복 실행하지 않는다. — 사용자 지시: "최종 모델은 70장 전체 학습" 방식을 lite23에 적용
  - 가설: E의 B4 추가 학습(#32)을 fold 학습 이미지(약 52장) 대신 train70 전체로 하면 데이터가 늘어 좋아질 수 있다. 반대로 4개 fold 모델이 비슷해져 앙상블 효과가 줄 수도 있다.
  - 레시피: 부모 model329 fold k에서 출발, `--train_splits train_fold{k},oof_fold{k}`(=70장), 2,000회, lr 3e-5, BN 고정, seed 819349. #32와 데이터만 다르다.
  - 출력: `/workspace/komap_model1/outputs/claude_b4_329_full70_bnfreeze/fold{1..4}`
  - 평가: OOF는 불가능하다(전부 학습에 씀). lite23의 E에서 B4를 이 4개로 바꾼 고정 후보 1개를 valid20(F 포함)으로 채점하고, lite23+B4ws(80.5464%) 대비 평균·시편 bootstrap·최악 이미지를 본다.
  - 채택 기준: lite23+B4ws보다 평균이 높고, 최악 이미지 손실이 0.1%p 이내.
  - Codex 제안: 결과가 좋으면 같은 방식(전체 70장, BN 고정 추가 학습)을 HR seed1 fold2·4, seed2 fold1·4에도 적용해 볼 수 있다(HR은 Codex 담당).

- **`claude_pkg_lite23_b4ws_20261010`** / 상태: **준비 완료, GPU6 대기 (16:1x KST)** — 사용자 지시: "lite23에 b4만 바꾼 구성으로 깃허브에 재현 zip 업로드"
  - 패키지: Codex lite23 ZIP(SHA f81b208c…)을 복사해 `Model349_fusion805_lite23_b4ws/`로 만들었다. 추가한 것은 `selected_config_b4ws.json`, `selected_assets_b4ws.json`, `parent_metadata/E/model329_ws`, `README.md`, `results_b4ws/`이다. `infer_expert.py`에는 `expert.b4_dir` 분기만 넣었고 기본 동작은 같다. Codex 원본 ZIP과 파일은 수정하지 않는다.
  - 검증: 서버 `/workspace/model349_compress/claude_pkg_lite23_b4ws_20261010/`에서 새로 푼 ZIP과 부모 fusion805(`19d56d4`)로 `infer_compact.py`를 실행한다(valid20, test10, R은 이전 원본 추론 재사용). 80.5464316%·0픽셀 차이가 확인되면 lhs에 새 ZIP으로 올린다.
  - GPU: 다른 GPU가 모두 사용 중이라, Claude의 lr1e4 평가가 끝난 뒤 같은 GPU6에서 순서대로 실행한다.
  - **사용자 지시로 검증 전에 lhs `1bdb4d8`에 업로드 (17:0x KST)**: `Model349_fusion805_lite23_b4ws_code.zip`(SHA 7c7a46c7…, `full70_experiment/` 포함). 서버 검증 폴더를 이 ZIP으로 교체했고, 끝나면 결과를 lhs README에 추가한다.

- **`claude_lite23_b4ws_s2warm_20261010`** / 상태: **완료 (15:4x KST)** — 조합 80.54955%(+0.0323 [−0.001, +0.069], 11/20, 최악 −0.074). B4ws 단독 80.54643%보다 +0.003뿐이고 구간이 0을 포함해, 단순한 B4ws 단독 후보를 유지한다. 결과 `claude_lite23_b4ws_20261010/combo_result.json`.
  - 목적: 서로 독립인 두 교체를 함께 넣은 lite23 고정 후보 1개를 채점한다. 두 교체는 Claude B4ws(#41, +0.029)와 Codex `accuracy81_lite23_seed2warm14_20261010`(HR seed2 fold1·4 추가 학습판, +0.006)이다.
  - 방법: Codex seed2warm14의 `expert/` 확률에서 0.5×원래 B4를 빼고 0.5×B4ws를 더한다. 이후 같은 F·채점을 적용한다. Codex 파일은 읽기만 하고 출력은 `claude_lite23_b4ws_20261010/combo_*`에 둔다.

- **`claude_b4ws_lr1e4_20261010`** / 상태: **완료 — 기각 (17:0x KST)**. OOF70 lr1e-4 79.5414 vs lr3e-5 79.4991(+0.042, fold 3/4 개선). 기준 +0.05 미달. `claude_b4_329_warmstart_bnfreeze_lr1e4/lr_eval.json`.
  - 가설: B4 BN 고정 추가 학습(#32, lr 3e-5)은 학습률이 낮아 이득이 덜 났을 수 있다. 학습률만 1e-4로 올린다.
  - 레시피: #32와 동일(부모 model329 fold k, train_fold k, 2,000회, BN 고정, seed 819349). 차이는 `--lr 1e-4`뿐이다. Codex HR lr1e-4 실험과는 모델이 다르다.
  - 출력: `/workspace/komap_model1/outputs/claude_b4_329_warmstart_bnfreeze_lr1e4/fold{1..4}`, GPU6에서 순서대로 실행.
  - 평가: oof_fold k에서 OOF70을 #32(lr 3e-5)와 비교. 공식 valid·test는 쓰지 않는다.
  - 다음 단계 조건: OOF70이 #32 대비 +0.05%p 이상이고 3개 이상의 fold에서 개선되면, lite23에 넣은 valid20 고정 후보 1개를 채점한다.

- **`claude_lite23_b4ws_20261010`** / 상태: **완료 — 후보 (2026-10-10 14:1x KST)**
  - 결과: 대조군 80.51727047%(Codex fresh lite23과 0픽셀 차이), **B4ws 교체 80.54643158% (+0.0292%p)**, 시편 bootstrap 95% [+0.0025, +0.0672], 14/20장 개선, 최악 −0.064%p. Codex gate 기준(평균 ≥ −0.01, 최악 ≥ −0.1) 통과.
  - 한계: E는 저장 확률 산술(−0.5·B4 + 0.5·B4ws)로 만든 값이다. 원본 영상부터의 전체 재추론은 아직 하지 않았다. B4ws 확률 자체는 원본 추론값이다.
  - 결과 `/workspace/model349_compress/claude_lite23_b4ws_20261010/result.json`, 마스크 `fusion_b4ws/`.
  - **원본 재추론 검증 등록 (14:2x KST, GPU6 유휴 확인; Codex round3의 GPU7, GPU0·3은 피함)**: Codex `infer_expert.py` 복사본(`infer_expert_ws.py`, B4 SHA만 B4ws로 교체)으로 valid20 E를 원본 영상에서 추론한 뒤, 같은 `infer_fusion.py`(F)와 `score_masks.py`로 채점한다. 저장 확률 결과와 픽셀 대조.
  - **원본 재추론 결과: valid20 80.54643158%, 저장 확률 후보와 0픽셀 차이.** E는 원본 영상 새 추론, R은 lite23과 같이 이전 원본 추론 라벨을 재사용했다. GPU6 반납. 출력 `claude_lite23_b4ws_20261010/fresh_expert`, `fresh_fusion`, `fresh_metrics.json`.
  - Codex에게: lite23 기준 후보에 넣으려면 E 추론에서 B4 4개를 `claude_b4_329_warmstart_bnfreeze/fold{1..4}/last.pt`로 바꾸면 된다(lhs `19d56d4` ZIP의 `--b4-warmstart`와 같은 가중치).
  - 목적: 사용자가 채택한 새 기준 lite23(80.51727047%)에서 **E의 B4-329 4 fold만 BN 고정 추가 학습판(#32)으로 바꾼 고정 후보 1개**를 채점한다. R14, HR seed1 fold2·4, seed2 fold1·4, 비율 0.5/0.25/0.25, F, 규칙은 모두 lite23 그대로다.
  - 방법: Codex `fresh23_gpu0/expert{0,1}`의 E 확률에서 0.5×원래 B4(FP16 가족 확률)를 빼고 0.5×B4ws(같은 FP16 규칙의 원본 추론 확률)를 더한다. 이후 lite23과 같은 `infer_fusion.py`(F 포함, CPU)와 `score_masks.py`로 채점한다. 대조군(교체 없음)이 80.51727%를 재현하는지 먼저 확인한다.
  - 데이터: 저장 확률만, 새 GPU 추론 없음. Codex 파일은 읽기만 하고 출력은 `/workspace/model349_compress/claude_lite23_b4ws_20261010/`에 둔다.
  - 판단: 고정 후보 하나이므로 선택 편향은 없지만 valid20 반복 사용의 한계는 남는다. lite23 대비 평균·최악 이미지·시편 bootstrap을 보고한다.

- **`claude_b4ws_seed2_20261010`** / 상태: **완료 — 기각 (13:2x KST). GPU 6·7 반납**
  - 결과 OOF70: 부모 79.1232, s1 79.4991, s2 79.4447, 평균 79.4872(**s1 대비 −0.012%p**, 34/70장 개선). fold별 평균−s1: +0.03/+0.01/−0.00/−0.06(fold1 계산값은 json 참조).
  - 해석: 시드를 바꿔도 추가 학습 이득은 재현된다(s2 +0.32%p vs 부모). 그러나 두 시드 평균은 이득이 없다. 같은 부모에서 출발한 짧은 학습이라 두 판이 거의 같기 때문이다.
  - 결과 `claude_b4_329_warmstart_bnfreeze_s2/ws2_eval.json`. valid20 채점은 하지 않았다(조건 미달).
  - 가설: E의 B4 추가 학습(#32)을 시드만 바꿔 한 번 더 하고 두 판을 평균하면, 시드 평균 효과로 B4가 더 좋아질 수 있다.
  - 레시피: #32와 동일(부모 model329 fold k, train_fold k, 2,000회, lr 3e-5, BN 고정, 결정적 worker). 차이는 `--seed 819350`뿐이다.
  - 출력: `/workspace/komap_model1/outputs/claude_b4_329_warmstart_bnfreeze_s2/fold{1..4}`. fold 1·3은 GPU6, 2·4는 GPU7에서 순서대로 돌린다.
  - 평가: oof_fold k(학습에 안 쓴 시편)에서 부모, ws(s1), ws(s2), 평균(s1+s2)의 OOF70을 비교한다. 공식 valid·test는 쓰지 않는다.
  - 다음 단계 조건: 평균이 ws(s1)보다 OOF70 +0.05%p 이상이고 3개 이상의 fold에서 개선되면, E′에 넣은 1단계 valid20을 고정 후보 하나로 채점한다.
  - Codex HR 추가 학습(GPU3)과는 모델이 달라 겹치지 않는다.

- **`claude_rule_eprime_20261010`** / 상태: **완료 — 현재 규칙 유지 (12:05 KST)**
  - 결과: {Al₃Ni, Al}이 16개 중 1위(80.5017). 다음은 {Al₃Ni} −0.051, {Al₃Ni, PSi, Al} −0.113%p. LOSO 19번 모두 같은 규칙을 골라 nested 차이 0. E′에서도 Eut 덮어쓰기는 손해. 결과 `claude_stage1_b4ws_fresh_20261010/rule_eprime.json`.
  - 가설: E′(B4 BN 고정 추가 학습)는 OOF에서 Eutectic Si가 +0.95%p 올랐다. 기존 E로는 Eut 덮어쓰기가 손해(#29)였지만, E′에서는 덮어쓰기 클래스 집합의 최적이 바뀔 수 있다.
  - 데이터: 원본 재추론 저장 결과만(R `accuracy81_stage1_fresh_20261009/runs/release_valid`, E′ `claude_stage1_b4ws_fresh_20261010/runs/expert_valid`). 새 추론 없음.
  - 탐색: 4클래스의 부분집합 16개 규칙 전체, nested LOSO(시편 하나를 빼고 규칙을 다시 선택).
  - Codex의 30개 규칙 선택(기존 E)·Claude #29(기존 E 쌍 규칙)와 차이: E′ 기준 재평가.
  - 채택 기준: nested가 현재 규칙({Al₃Ni, Al}) 대비 +0.05%p 이상이고 하한 0 이상.

- **`claude_r_b4ws_gamma_20261010`** / 상태: **완료 — 기각 (2026-10-10 11:08 KST)**
  - 결과 (기준 = 현재 최선 80.50171%):
    - (b) 교체 없이 재선택: 전체 valid로 고르면 80.5037%(+0.002)이지만 **nested −0.212%p [−0.498, −0.037]**, 최악 이미지 −2.5%p. valid20으로 비율을 다시 고르는 것 자체가 과적합이다.
    - (a) B4 7개 교체 + 재선택: 전체 valid로 고르면 80.5054%(+0.004), **nested −0.036%p [−0.227, +0.115]**, 최악 −1.2%p. 교체만 할 때(80.4694, 고정 비율)의 손해는 비율 재선택으로 거의 메워지지만 이득으로 바뀌지 않는다.
  - 결론: R의 B4 추가 학습판은 valid20에서 비율을 다시 맞춰도 쓸 수 없다. 남은 길은 팀원 재학습의 OOF 선택(oof_design)에 BN 고정 추가 학습 단계를 넣는 것뿐이다. 이 방향(R 결합 비율의 valid 재선택)은 반복하지 않는다.
  - 결과 파일: `claude_r_b4ws/gamma_ws.out`, `gamma_ws.json`.
  - 가설: #33·#36에서 R의 B4 7개를 BN 고정 추가 학습판으로 바꾸면 단독은 +0.33%p인데 결합은 손해였다. exh14 결합 비율이 원래 B4에 맞춰져 있기 때문으로 보고, B4 관련 비율을 다시 고르면 이득이 살아나는지 본다.
  - 범위의 한계: R의 HRNet336/347·269·고해상도 경로는 train70 전체로 학습해 OOF가 없다. 그래서 비율을 train70 OOF로 고를 수 없고, valid20에서 **시편 하나를 빼고 선택 전체를 다시 하는 nested LOSO**로만 평가한다. 진짜 OOF 선택은 팀원 재학습(oof_design) 범위다.
  - 탐색: B4 관련 8개 비율(primary·al3ni(326), al(322), gamma329 4개)만 좌표 상승(최대 3회, 이득 1e-6 초과일 때만 이동). 두 군: (a) B4 7개 교체 + 재선택, (b) 교체 없이 재선택(대조군; 비율 재선택 자체의 효과). 목표 함수는 1단계(E′=B4ws) 점수. 기준은 현재 최선 80.5017%.
  - 데이터: 저장된 valid20 확률만(`claude_r_b4ws/valid_probs`, lite14 probe 캐시). 새 학습·추론 없음. test 미사용.
  - 경로: `/workspace/komap_model1/outputs/claude_r_b4ws/gamma_ws.py` → `gamma_ws.out`, `gamma_ws.json`
  - 채택 기준: nested가 기준 대비 +0.05%p 이상이고 신뢰구간 하한이 0 이상.
- **`claude_pkg_b4ws_20261010`** / 상태: **완료 — lhs `19d56d4` 업로드 (2026-10-10 11:5x KST). GPU4 반납**
  - 검증: 새 폴더에 푼 ZIP으로 E′ 확률 20/20 SHA 일치, 1단계 라벨 0픽셀 차이, valid20 80.50167745%. 이전 ZIP 대비 추가 11개, 변경 3개(README, SOURCE_SHA256, `infer_model401.py`의 선택 옵션)뿐이며 Codex 파일은 그대로다.
  - 목적: #34 후보(E의 B4-329를 BN 고정 추가 학습판으로 교체한 1단계, 80.5017%)를 fusion805 ZIP에 추가한다. lhs 최신 커밋 `619500d`의 ZIP을 기준으로 하고, Codex 파일은 바꾸지 않는다(`infer_model401.py`에 선택 옵션만 추가).
  - 검증: ZIP을 새 폴더에 풀어 GPU4에서 valid20 E′ 추론 → 1단계 → 채점. 80.50167745%와 라벨 SHA 일치 확인.

- **`claude_grow_e_20261009`** / 상태: **완료 — 기각 (2026-10-09 19:0x KST)**
  - 결과: 전체 valid로 구성원을 하나씩 더하는 탐색에서는 아무것도 추가하지 않는 것이 최선이었다(80.4744%). 사후 최고는 N3 + 릴리스 329 추가 80.5058%(+0.031%p)였다. **nested LOSO는 1단계 대비 −0.072%p [−0.125, −0.026]**로 명확한 손해다.
  - 결과 파일: `/workspace/model349_compress/experts/grow_e.out`, `grow_e_result.json`.
  - 결론: 저장된 다른 계열 확률을 E에 평균해 Al₃Ni 전문가를 강화해도 일반화되지 않는다. 같은 구성원 풀로 이 방향을 반복하지 않는다.
- **`claude_pair_rule_20261009`** / 상태: **완료 — 기각 (2026-10-09 19:15 KST)**
  - 결과: 4,096개 쌍 규칙 중 **현재 1단계 규칙이 정확히 1위**(80.4744%)이고, 쌍 하나를 바꾸면 모두 손해다.
    - 쌍별 기여(1단계에서 해당 쌍을 끌 때의 손실): Eut→Al₃Ni −0.12, Al→Al₃Ni −0.05, PSi→Al₃Ni −0.04%p.
    - E의 Eutectic Si를 따르면 손해다(Al₃Ni→Eut −0.82%p).
  - nested −0.160%p [−0.383, −0.008]: 시편을 뺄 때 다른 규칙이 뽑혀서 생기는 손해다.
  - 결론: 같은 R·E로 규칙만 다듬어서는 더 오르지 않는다.
  - 결과 파일: `experts/pair_rule.out`, `pair_rule_result.json`.
- **`claude_e_b4gray_w4000_20261009`** / 상태: **완료 — 기각 (2026-10-09 23:55 KST)**
  - 결과는 목록 #30에 있다. fold 1~3로 한 잠정 평가에서 nested +0.005%p였고, fold 4 학습은 중지했다. Claude 소유 GPU 프로세스는 없다.
- **`claude_b4_329_warmstart_20261010`** / 상태: **완료 — 기각 (2026-10-10 01:00 KST, GPU 반납)**
  - 결과: 시편 보류 OOF70 부모 79.1232% → 추가 학습 79.0190%, **−0.1042%p**(29/70장 개선).
    - fold별: +0.07, −0.21, −0.03, −0.25%p.
    - 클래스별(Al, Al₃Ni, PSi, Eut): −0.05, −0.10, −0.07, −0.19%p.
  - 결론: 조건(+0.1%p, 3개 fold 개선)을 넘지 못해 E에 넣는 valid 평가는 하지 않는다.
  - Codex HRNet 레시피와의 차이: BN 고정 없음. B4-329는 원래 16,000회로 이미 수렴했을 가능성도 있다. 같은 레시피로 B4 추가 학습을 반복하지 않는다.
  - 결과 파일: `/workspace/komap_model1/outputs/claude_b4_329_warmstart/oof_compare.json`, `run.log`.
- **`claude_b4_329_warmstart_bnfreeze_20261010`** / 상태: **학습·OOF 평가 완료 — 조건 통과 (2026-10-10 02:30 KST)**
  - 결과: 시편 보류 OOF70 부모 79.1232% → BN 고정 추가 학습 **79.4991%, +0.376%p**(51/70장 개선).
    - fold별: +0.72, +0.21, −0.06, +0.62%p.
    - 클래스별(Al, Al₃Ni, PSi, Eut): +0.00, +0.22, +0.33, **+0.95%p**.
  - BN 고정 없는 같은 레시피는 −0.104%p였으므로, 차이를 만든 것은 BN 고정이다. Codex HRNet 추가 학습 결과와 같은 방향이다.
  - 다음 단계(실행 전 등록): **`claude_e_b4ws_stage1_valid_20261010`**
    - E의 B4-329 4개를 이 추가 학습 모델 4개로 바꾼 E′ = 0.5·B4ws(D4) + 0.25·HR392 시드1 + 0.25·시드2를 만든다(HR은 기존 그대로).
    - 1단계 규칙 {Al₃Ni, Al}로 R(exh14)을 덮어써 valid20을 채점한다.
    - 고정 후보 하나라 선택이 없다. 1단계 80.4744% 대비 차이와 시편 bootstrap 구간을 기록한다. 독립 PNG 채점 전까지는 확정 점수가 아니다.
    - Codex의 HR392 추가 학습 결과는 나오면 별도로 조합한다. Codex 담당이며 Claude는 HR을 재학습하지 않는다.
    - **결과 (저장 확률 재조합, 2026-10-10 02:45 KST):** 1단계 80.4744% → **80.5017%, +0.0273%p**.
      - 시편 bootstrap [+0.0057, +0.0559], 14/20장 개선, 최악 A125 −0.048%p.
      - 단독 성능: B4ws 79.163%(B4-329 78.833% 대비 +0.33, Eut +0.81), E′ 79.530%.
      - 고정 후보 하나라 선택 편향은 없다. 원본부터 재추론한 독립 채점은 아직 하지 않았다.
      - 경로: `/workspace/komap_model1/outputs/claude_b4_329_warmstart_bnfreeze/valid_stage1/` (`per_image.json`, labels, probs).
- **`claude_e_b4ws_rule_20261010`** / 상태: **완료 — 원래 규칙 유지**
  - E′로 16가지 클래스 집합을 다시 골라도 {Al₃Ni, Al}이 1위였다(nested 차이 0). E′의 Eutectic Si 덮어쓰기는 여전히 손해다.
- **`claude_r_b4ws_20261010`** / 상태: **완료 — 결합 단계 기각 (2026-10-10 05:00 KST, GPU 반납)**
  - **모델별 OOF:** BN 고정 추가 학습이 7개 중 6개를 개선했고, 평균 78.865% → 79.198%(**+0.333%p**)였다.
    - 모델별 변화: 322 f3 +0.09, f4 +0.85 / 326 f1 +0.33, f2 +0.40 / 329 f1 +0.41, f3 −0.12, f4 +0.36%p.
    - 분할 확인: 부모가 학습 fold 이미지에서 85~87%, OOF에서 76~84%로, 같은 분할과 맞는다.
  - **valid20 결합 (exh14 결합 비율 고정, 저장 확률):**
    - R: 80.2309% → 80.2130%
    - 1단계(E′): 80.5017% → 80.4694%(**−0.032%p**, [−0.25, +0.14], 최악 −1.62%p)
  - 결론: 개별 모델은 좋아졌지만 릴리스 가중치에 맞춘 결합 비율과 맞지 않는다. 결합 비율을 다시 맞추려면 OOF 기반 선택이 필요하다. valid 재선택은 하지 않는다.
  - 결과 파일: `/workspace/komap_model1/outputs/claude_r_b4ws/eval_result.json`.
- **현재 Claude 최선 후보:** 1단계, E′ = 0.5·B4-329ws(BN 고정 추가 학습, D4) + 0.25·HR392 시드1 + 0.25·시드2, R = 릴리스 exh14.
  - valid20 **80.5017%**(저장 확률, 1단계 대비 +0.027%p [+0.006, +0.056]). 원본 재추론과 test는 아직 하지 않았다.
  - 다음 결합 제안: Codex의 HR392 추가 학습 4 fold가 끝나면 E″ = B4ws + HRws로 1단계를 평가한다. Codex가 HR 결과 경로를 기록하면 Claude가 CPU 재조합만 한다(중복 학습 없음).
- **`claude_stage1_b4ws_fresh_20261010`** / 상태: **완료 (2026-10-10 06:00 KST, GPU 반납)**
  - valid20: 원본 재추론과 독립 PNG/GT 채점 결과 **80.50167744815%**다. 저장 확률 재조합 라벨과 **0픽셀 차이**다. 기존 1단계 80.4744% 대비 +0.027%p.
  - test10: 새 1단계는 기존 1단계와 15,335/9,074,964픽셀(0.169%) 다르고, R과는 1.80% 다르다. test GT는 없다.
  - 산출물: `runs/{expert_valid,expert_test,stage1_valid,stage1_test}`, `runs/stage1_valid/metrics.json`.
  - 한계: B4ws는 train70 fold 데이터로만 학습했다. 다만 이 레시피 선택은 OOF70 기준으로 했고, valid는 1회 채점에만 썼다. 1단계 규칙과 E 구성은 원래 valid로 고른 것이라 그 편향은 남는다.
  - 목적: 위 최선 후보의 원본 이미지 재추론 검증. valid20은 독립 PNG/GT 채점, test10은 마스크를 만들고 기존 1단계와 비교한다.
  - 방법:
    - Codex `accuracy81_stage1_fresh_20261009`의 `infer_model401.py`·`infer_stage1.py`를 복사해 쓰고, B4 4개 경로만 `claude_b4_329_warmstart_bnfreeze/fold{1..4}`로 바꾼다. 새 B4는 manifest에 없으므로 SHA 검사만 그 4개에 대해 끈다.
    - HR392 8개는 원래 자산을 그대로 쓴다.
    - R은 Codex의 원본 재추론 결과(`runs/release_valid`, `release_test`)를 재사용한다.
  - 경로: 서버 `/workspace/model349_compress/claude_stage1_b4ws_fresh_20261010/`. Codex의 원본 폴더는 수정하지 않는다.
  - 가설: BN 고정 추가 학습이 E의 B4-329를 개선했다(#32 OOF70 +0.376%p). R의 B4 7개(322 f3·f4, 326 f1·f2, 329 f1·f3·f4, 릴리스 가중치)도 같은 방법으로 개선될 수 있다.
  - 레시피: #32와 같다(BN 고정, 2,000회, lr 3e-5, seed 819349). 각 모델의 원래 입력·배율·샘플링을 유지하고, `data_model322_group_cv` train_fold k만 사용한다.
  - 평가:
    1. fold별 부모 대비 OOF를 비교한다. 이때 부모가 train_fold 이미지와 oof_fold 이미지에서 내는 성능 차이로 릴리스 가중치의 분할이 같은지도 확인한다.
    2. R의 7개를 바꾼 뒤 exh14 결합(결합 비율 고정)과 1단계 valid20을 저장 확률로 재조합한다. 1단계 E는 #32의 E′를 쓴다.
  - 경로: 서버 `/workspace/komap_model1/outputs/claude_r_b4ws/`
  - 가설: 바로 위 실험의 실패는 BN 통계가 약 52장 소량 데이터로 다시 학습되며 흔들린 탓일 수 있다. Codex HRNet 레시피처럼 BN 통계를 고정하면 B4도 개선될 수 있다.
  - 바로 위 실험과의 차이: BN 고정 하나뿐이다. 학습 코드 사본 `train_bnfreeze.py`에서 학습 시작 시 모든 BatchNorm을 eval 모드로 둔다. 데이터, 부모, 2,000회, lr 3e-5, seed, 평가 방식은 같다.
  - 경로: 서버 `/workspace/komap_model1/outputs/claude_b4_329_warmstart_bnfreeze/`
  - 다음 단계 조건: 위와 같다(OOF70 +0.1%p 이상, 3개 이상 fold 개선).
  - 가설: Codex `accuracy81_HR_warmstart_*`에서 E의 HRNet392를 같은 fold 학습 데이터로 저학습률 추가 학습하자 보류 fold가 +0.59%p 올랐다. E의 나머지 절반인 B4-329에도 같은 효과가 있는지 본다.
  - 역할 분담: HRNet은 Codex가 담당한다. Claude는 B4-329만 다루며, 같은 HRNet 실험은 하지 않는다.
  - 데이터·부모:
    - 데이터: `data_model322_group_cv` train_fold k에서만 학습하고, oof_fold k(학습에 안 쓴 시편)로 평가한다. 공식 valid·test는 쓰지 않는다.
    - 부모: `model329_fullstats_width4000_uniform_group_oof/fold{k}/last.pt`
  - 레시피: 329와 같은 입력·배율(zcdf3, image_global, target_width 4000, uniform). 차이는 `--init_checkpoint` 부모, `--iters 2000`, `--lr 3e-5`, `--seed 819349`, `--deterministic_workers`뿐이다. BN 고정 옵션은 이 학습 코드에 없어 쓰지 않는다(Codex HR 레시피와의 차이).
  - 평가:
    - fold마다 부모와 추가 학습본을 같은 코드(D4 TTA)로 oof_fold k 이미지에 추론하고, 시편 보류 mIoU를 비교한다.
    - 4개 fold를 합친 OOF70 변화도 계산한다.
  - 경로: 서버 `/workspace/komap_model1/outputs/claude_b4_329_warmstart/`
  - 다음 단계 조건: OOF70이 +0.1%p 이상이고 3개 이상의 fold에서 개선되면, E에 넣어 1단계 valid 평가를 진행한다.
  - fold1→fold4는 GPU4에서 차례로, fold2는 GPU6, fold3는 GPU7. 반복당 약 1.03초, 완료 예상은 10-10 04:40 KST 전후.
  - 서버 `model326` 학습 코드에는 `--deterministic_workers`가 없어 첫 실행이 실패했다. lhs 패키지에 넣은 같은 계열의 최신 코드(`oof_smoke/pkg/.../vendor/experiments/model326_hybrid_fullstats/KoMaP/train.py`)와 결정성 런타임으로 다시 시작했다.
  - 실행기 `/workspace/komap_model1/outputs/claude_b4gray_w4000_run.sh`, 평가 `.../claude_b4gray_w4000/eval_b4gray.py`.
  - 가설: E의 Al₃Ni 이득은 서로 다른 폭 정규화 모델(B4 zcdf + HRNet)을 평균한 데서 나온다. 입력만 다른 세 번째 폭 정규화 계열(회색조 B4)을 E에 더하면 Al₃Ni 판단이 더 좋아질 수 있다.
  - 기존 작업과의 차이:
    - #28에서 더한 릴리스 322(회색조)는 폭 정규화 없이 train70 전체로 학습한 모델이다. 이번에는 폭 정규화와 E와 같은 4-fold 분할을 쓴다.
    - Codex의 HRNet 질감 입력(GPU 3)과는 모델 계열·입력이 다르다.
  - 데이터·레시피:
    - 데이터: `data_model322_group_cv` train_fold1~4 (E와 같은 시편 분할, train70만 사용). valid·test는 학습에 쓰지 않는다.
    - 학습: `model326_hybrid_fullstats/KoMaP/train.py`, `--norm imagenet --input gray3 --scale 1 --target_width 4000 --sample uniform --iters 16000 --seed 0 --deterministic_workers`. 나머지는 329와 같다.
  - 출력: `/workspace/komap_model1/outputs/claude_b4gray_w4000/fold{1..4}`
  - 평가 계획 (모두 고정 후보 소수):
    - valid20에서 4 fold 평균 D4 확률을 만든다.
    - E' 2종을 만든다: B4를 (329 + 새 B4) 평균으로 교체, 또는 셋을 1/3씩.
    - 1단계 규칙 {Al₃Ni, Al}로 R을 덮어쓴다. 1단계 대비 점수와 nested LOSO를 기록한다.
    - 채택 기준: nested +0.05%p 이상이고 신뢰구간 하한이 0 이상이다.
  - 가설: 1단계는 E의 예측 클래스만 보고 덮어쓴다. R과 E가 다르게 예측한 조합(R클래스→E클래스, 12가지)마다 E를 따를지 따로 정하면 더 정확할 수 있다.
  - 기존 작업과의 차이: Codex의 30개 규칙(클래스 집합 단위)이나 Claude #25(1단계 위 다른 전문가)와 달리, R과 E 두 라벨의 쌍별 전이 규칙이다.
  - 데이터/선택 규칙: 원본부터 재추론한 R(`release_valid`)과 E(`expert_valid`) 저장 결과만 쓴다(valid20, 새 추론 없음). 12개 쌍을 켜고 끄는 2^12=4,096 조합 전체를 채점하고, nested LOSO로 선택까지 포함해 평가한다.
  - 경로: `/workspace/model349_compress/experts/pair_rule.py` → `pair_rule.out`
  - 채택 기준: nested가 1단계 대비 +0.05%p 이상이고 신뢰구간 하한이 0 이상일 때만 후보로 올린다.
  - 가설: fusion805 1단계 이득은 E의 Al₃Ni 판단에서 나온다. R 위에 덮어쓸 때 E 전체(Al₃Ni) +0.205%p, HRNet392만 +0.106%p, B4-329만 −0.078%p였다. 서로 다른 계열을 평균한 효과로 보인다. 이미 저장된 다른 계열 확률(릴리스·P100 구성원, N1~N4, PL1/PL4, 고해상도 기본 경로)을 E에 더해 Al₃Ni 전문가를 강화하면 덮어쓰기 이득이 커지는지 본다.
  - 기존 작업과의 차이: Codex `p100_mask_consensus`(마스크 합의), `hr392_complement`(R 경로에 확률로 혼합)와 달리, E의 **확률 평균에 구성원을 추가**하고 같은 1단계 덮어쓰기 규칙을 적용한다.
  - 데이터/선택 규칙: 저장된 valid20 확률만 사용한다(새 추론 없음). E' = (2·E + 추가 구성원 합)/(2+k). 규칙은 {Al₃Ni} 또는 {Al₃Ni, Al}. 구성원을 최대 4개까지 하나씩 추가하는 탐색이다. **시편 하나를 빼고 탐색 전체를 다시 하는 nested LOSO**로 평가한다.
  - 서버 경로: `/workspace/model349_compress/experts/grow_e.py` → `grow_e.out`
  - 채택 기준: nested LOSO가 1단계 대비 +0.05%p 이상이고 신뢰구간 하한이 0 이상일 때만 원본 재추론 후보로 올린다.

## Claude 전체 실험·검증 목록 (2026-10-07 ~ 10-09)

- **기준**:
  - 지표는 공식 image-wise 4-class mIoU(빈 union = 0)이고, 별도 표기가 없으면 valid20 기준이다.
  - "nested"는 시편 하나를 빼고 선택 과정 전체를 다시 한 값이다(LOSO, 19그룹).
  - "저장 확률"은 원본 재추론이 아닌 캐시 재생이다.
- **서버 경로**: 서버 경로의 기본 위치는 `/workspace/model349_compress/`이다.
- **같은 조건 반복 금지**: 아래와 같은 조건(데이터·가중치·탐색 공간)의 실험은 다시 실행하지 않는다.

| # | 실험 (서버 경로) | 내용 | 결과 | 판단 |
|---|---|---|---|---|
| 1 | `model349_release_verify` | model346 릴리스 가중치 23개 P100 재추론 | valid 80.0683%, 팀원 환경 80.0688%와 일치 | 재현 확인 |
| 2 | `probe`, `probe_high`, `results/eval*.json` | 경량화 탐색 | TTA 1-view·기본 경로 경량화는 손실 없음. B4 soup −0.45%p. 증류는 단일망 76.9~78.7%로 기각 | lite 계열 근거 |
| 3 | `exhaustive`, 로컬 `exhaustive_20261008/` | 구성 요소 부분집합 8,093개 전수 탐색 (저장 확률, 1-view) | 최고 80.2310% → **exh14**. nested +0.092%p [−0.12, +0.39]. base12식 326 fold 선택 절차는 nested −0.137%p | exh14 채택 (제출 후보) |
| 4 | `Model349_exh14` | exh14 원본부터 재추론 | valid 80.2312313%, test10은 오프라인 대비 34픽셀 차이 | lhs `1704fff` |
| 5 | `test_compare_20261008` | test10 마스크 비교 (Model349 기준) | 일치율 exh14 99.38%, base12 99.31% | 참고 |
| 6 | `tune` (T) | B4/HRNet/269 TTA 시점 24조합 | 최고 80.271%, nested −0.022%p | 기각 |
| 7 | `tune` (G) | 결합 비율 좌표 상승 | 4단계에서 +0.015%p, 비용 대비 무의미해 중단 | 중단 |
| 8 | `probe_scale` | 구성원 배율 변경 (전체 적용, 원래 배율과 평균 포함) 26종 | nested −0.034%p | 기각 |
| 9 | (오류 분해) | exh14 오류 분석 | 오류의 92%가 GT 경계 2px 안. 소수상 경계 1px 확장은 −4.6%p | 진단 |
| 10 | `eut_spec`, `komap_model1/outputs/model_eut_{L1,L2,P1,P2}` | Eutectic Si 전용 HRNet 4종 (336에서 출발, Lovász·경계 가중) | 단독 최고 78.25% (L2). 추가 시 nested −0.11~−0.30%p. 336 자리 교체 −0.20%p | 기각 |
| 11 | `feat_refine` | 손수 만든 질감 특징 33개 + LightGBM 픽셀 보정 (LOSO) | 앙상블 특징만 −0.73%p, 질감 추가 −0.89%p. 질감 특징의 순효과 −0.16%p | 기각 |
| 12 | (메타데이터) | step·이미지 크기·같은 시편 이미지·PNG 메타데이터 분석, Al/Eut 편향 LOSO | 쓸 정보 없음. 편향 nested −0.08%p | 기각 |
| 13 | `overlap_check.py` | 100장 SIFT 겹침 검사 | 겹침 없음 | 진단 |
| 14 | (GT 정렬) | GT 위치 어긋남·해상도·밝기 임계값 재현 여부 | 어긋남 없음, 원 해상도, 밝기만으로 최대 88% | 진단 |
| 15 | `scale_adapt.py`, `coarse_valid` | 이미지별 배율 선택. 굵은 조직 합성셋 (crop × 1.6) | 이미지별 정답 상한 +0.17%p. 실제 이미지 축소는 −4.7~−7.3%p. 합성셋 +3.8%p는 흐림 때문 | 기각 |
| 16 | (B4 OOF 분석) | 굵거나 흐린 train 이미지의 OOF 성능 | 오히려 쉬움 (83.2 vs 78.0%) | 진단 |
| 17 | `sp_em.py`, `probe_ctx` | SLIC 슈퍼픽셀, label-shift EM, HRNet 768/1024 창 | −0.63%p / −1.32%p / +0.007%p | 기각 |
| 18 | `probe_tent` | B4·HRNet 구성원에 TENT | nested +0.028%p [−0.022, +0.076] | 기각 |
| 19 | (잡음 제거) | 1~9px 점 잡음 제거 | nested −0.016%p | 기각 |
| 20 | `newmem`, `komap_model1/outputs/{model412_b4_scale4_train70, model413_hrnet_scale3_ft, model414_hrnet_zcdf_ft, nm2_convnext_tiny_326recipe}` | 새 구성원 N1 HRNet 3배, N2 ConvNeXt-T, N3 HRNet zcdf, N4 B4 4배 (train70) | 단독 76.1~78.1%. 추가 시 nested −0.16~0%p | 기각 |
| 21 | `komap_model1/outputs/pl{1,2,3,4}_*` | 자기학습 (train70 + valid·test 이미지에 exh14 의사 정답). PL4는 릴리스 336에서 출발 | 단독 77.1~79.3% (증류 효과). 추가 시 nested −0.10~−0.03%p | 기각 |
| 22 | `dual_eval.py` | 릴리스판·P100판 가중치를 구성 요소별 평균 | 전체 평균 −0.17%p, 347만 P100판 +0.02%p, nested +0.020%p | 기각 (미미) |
| 23 | `probe_p100*`, `p100_bundle`, `exhaustive_p100` | P100판 23개로 전수 탐색 재현 | exh14 조합을 이식하면 P100 FULL 대비 +0.11%p (오프라인 1-view). P100 자체 탐색 nested −0.082%p | 조합 이득 재현, 탐색 절차는 불안정 |
| 24 | (검토) | fusion805 문서·ZIP 검토 | 1단계 권장, 2단계 비권장 | lhs `aa418cb`에 반영 |
| 25 | `experts/rules.py` | 1단계 위에 다른 전문가 덮어쓰기·두 전문가 일치 규칙 289개 | nested −0.215%p | 기각 |
| 26 | `experts/decompose_e.py` | E 분해 (R 위 덮어쓰기) | E Al₃Ni+Al +0.243, E Al₃Ni +0.205, HR392 Al₃Ni +0.106, B4-329 Al₃Ni −0.078%p | 진단 → #27 |
| 27 | `oof_smoke`, lhs `aa418cb` → `619500d` | OOF fold × 시드 설계 코드와 smoke | 학습 11/19단계 통과 후 인계 | 팀원 인계 |
| — | lhs `619500d` | **사고 복구:** `aa418cb`에서 fusion805 ZIP을 Codex `d4ef4b0`/`7e246b5`보다 오래된 사본으로 다시 만들어 Codex의 갱신(`infer_stage1.py`, 1단계 원본 재추론 결과, README·decision)을 덮어썼다. `7e246b5` ZIP을 기준으로 다시 만들고 `oof_design`·`oof_expert`·README 절만 추가했다. `--stage1-only` 수정은 빼고 `infer_stage1.py`를 사용한다. | 복구 완료 | 앞으로 ZIP을 갱신하기 전에 항상 최신 커밋의 ZIP을 기준으로 한다 |
| 28 | `experts/grow_e.py` (`claude_grow_e_20261009`) | E에 저장된 다른 계열 확률을 하나씩 추가 + 1단계 덮어쓰기 | 추가하지 않는 것이 최선. nested −0.072%p [−0.125, −0.026] | 기각 |
| 29 | `experts/pair_rule.py` (`claude_pair_rule_20261009`) | R·E 불일치 12쌍별 따를 모델 선택 (규칙 4,096개) | 1단계 규칙이 1위. nested −0.160%p | 기각, 1단계 규칙 최적 확인 |
| 30 | `komap_model1/outputs/claude_b4gray_w4000` (`claude_e_b4gray_w4000_20261009`) | 폭 정규화 + 회색조 B4를 E와 같은 4-fold 분할로 학습해 E에 추가 (fold 1~3, D4 TTA) | 단독 79.03%(329보다 +0.19). E'a 80.479%(+0.005), E'b 80.437%(−0.037). nested +0.005%p [−0.012, +0.023] | 기각. fold 4는 결론이 바뀔 가능성이 낮아 학습 중(23:55 KST) 중지 |
| 31 | `komap_model1/outputs/claude_b4_329_warmstart` (`claude_b4_329_warmstart_20261010`) | E의 B4-329 fold 모델을 같은 fold 학습 데이터로 2,000회 추가 학습 (lr 3e-5, BN 고정 없음) | 시편 보류 OOF70 −0.104%p (fold 4개 중 1개만 개선) | 기각 |
| 32 | `komap_model1/outputs/claude_b4_329_warmstart_bnfreeze` | #31과 같은 조건에 **BN 고정** | OOF70 **+0.376%p** (51/70, Eut +0.95). E에 넣은 1단계 valid20 **80.5017%** (+0.027 [+0.006, +0.056]). 덮어쓰기 규칙은 {Al₃Ni, Al} 유지 | **채택 후보** — 원본 재추론 80.50167745% 확인 (#34) |
| 34 | `model349_compress/claude_stage1_b4ws_fresh_20261010` | #32 후보를 원본부터 재추론 (valid20 + test10) | valid **80.50167745%** (재조합과 0픽셀 차이). test는 기존 1단계와 0.17% 차이 | 현재 Claude 최선 |
| 36 | `komap_model1/outputs/claude_r_b4ws/subset_swap.py` | R의 B4 계열(322/326/329)별로 BN 고정 추가 학습판 교체 여부 2³=8조합 + 1단계(E′=B4ws). 결합 비율 고정 | 교체 안 함이 사실상 최선(326만 +0.003). 329 교체는 −0.03%p(최악 −1.4). nested −0.096%p [−0.28, −0.00] | 기각. R 쪽 B4 교체 방향 종료 |
| 37 | `komap_model1/outputs/claude_r_b4ws/gamma_ws.py` (`claude_r_b4ws_gamma_20261010`) | R의 B4 관련 비율 7개를 valid20에서 다시 선택 (B4 교체 / 미교체), 1단계 목표, nested LOSO | 미교체 nested −0.212%p [−0.50, −0.04], 교체 nested −0.036%p [−0.23, +0.11] | 기각. valid 비율 재선택 자체가 과적합 |
| 38 | `model349_compress/claude_pkg_b4ws_20261010`, lhs `19d56d4` | #34 후보를 fusion805 ZIP에 추가하고 새 폴더에서 재현 | 0픽셀 차이, 80.50167745% | 업로드 완료 |
| 39 | `claude_stage1_b4ws_fresh_20261010/rule_ep.py` | E′ 기준 덮어쓰기 클래스 16규칙, nested | {Al₃Ni, Al} 1위, LOSO 19회 모두 동일 | 현재 규칙 유지 |
| 40 | `komap_model1/outputs/claude_b4_329_warmstart_bnfreeze_s2` | B4 추가 학습 시드 2 + 두 시드 평균 (OOF70) | s2 단독 +0.32%p(부모 대비, 이득 재현), 평균은 s1 대비 −0.012%p | 기각 |
| 41 | `model349_compress/claude_lite23_b4ws_20261010` | 새 기준 lite23에서 E의 B4만 B4ws로 교체 (저장 확률, F 포함) | **80.5464%** (+0.029 [+0.003, +0.067], 14/20, 최악 −0.064) | 후보. E 원본 재추론으로 0픽셀 차이 확인 |
| 42 | `claude_lite23_b4ws_20261010/combo.py` | lite23 + B4ws + Codex HR seed2 warm fold1·4 | 80.5496% (+0.032 [−0.001, +0.069]) | B4ws 단독 대비 +0.003, 미채택 |
| 43 | `model349_compress/claude_team_newE_20261010` | 팀원 OOF 재학습 E(seed 0, 4 fold) valid 라벨을 원래 E·릴리스 R과 비교 (사용자 제공 TEMP.zip) | 새 E 단독 79.295 vs 원래 E 79.418 (픽셀 99.36% 일치). 릴리스 R 위 1단계: 새 E 80.462 vs 원래 E 80.474 (−0.013 [−0.062, +0.032]). 릴리스 R에 [Al₃Ni, Eut] 규칙은 79.351(−1.12) | 재학습 손실은 거의 전부 R 쪽. E 재학습은 정상 |
| 44 | (로컬 분석, 사용자 제공 `model349_newweights_valid_analysis.tar.gz`) | 팀원 OOF 재학습 R(seed 0, fold 평균)에 exh14 기본 설정 vs OOF 선택 설정 | 기본 설정 R **80.02**(릴리스 80.23 대비 −0.21), OOF 선택 설정 79.29(기본 대비 −0.74, A71 −8.5·A11_step1 −4.0, A71 제외 −0.33, 8/20 개선, 시편 bootstrap [−1.75, +0.02]). 기본 R + {Al₃Ni, Al} 80.23(+0.20). OOF에서는 선택 설정이 +0.39(outer +0.16) | 재학습 하락의 대부분은 가중치가 아니라 OOF 설정 재선택. fold 평균 가중치 자체는 −0.21로 이전 재학습(80.05)과 비슷 |
| 45 | `komap_model1/outputs/claude_b4_329_warmstart_bnfreeze_lr1e4` | B4 추가 학습 lr 1e-4 (OOF70) | +0.042%p vs lr 3e-5 | 기각(기준 미달) |
| 46 | `model349_compress/claude_refiner_retrain_20261010` | F 재학습 스크립트 이식(`refiner_retrain/`): 원래 E OOF로 학습 입력 재생성 → F 재학습 | 입력 70장 배열 동일, E OOF70 79.98586 재현, **F 가중치 SHA 동일(932606f8…)** | 사용자 지시 B: lite23 ZIP에 시드별 전체 재학습(`retrain/`) 추가 준비 |
| 47 | `model349_compress/claude_lite23_eratio_20261010` | lite23+B4ws의 E 비율 B4 w∈{.4,.5,.6,.7} (F 포함, nested) | 0.4·0.5 거의 같음, 0.7 −0.08. nested −0.014 | 기각, 0.5 유지 |
| 35 | `komap_model1/outputs/claude_b4_329_warmstart_bnfreeze/e2_valid.py` | E″ = B4ws + Codex HRws 시드1(`accuracy81_HR_warmstart_valid20_20261009` fold 확률) + HR 시드2, 1단계 (고정 후보, 저장 확률) | B4ws+HRws 80.5042% (+0.030 [−0.003, +0.065]). B4ws만 80.5017%, HRws만 80.4826%. E 단독 79.57% | HR 추가분은 1단계에서 거의 효과 없음(+0.0025). B4ws 단독 후보 유지 |
| 33 | `komap_model1/outputs/claude_r_b4ws` | R의 B4 7개에 BN 고정 추가 학습 | 모델 OOF 평균 +0.333%p. exh14 결합 비율 고정 시 R −0.018, 1단계 −0.032%p | 결합 단계 기각 (결합 비율 불일치) |
| — | (사용자 보고, 팀원 환경) | fusion805 OOF 설계 적용 전 버전을 재학습 | 재학습 R 80.051924%, 1단계 80.236507%(+0.185), 2단계 80.267268%(1단계 대비 +0.031) | 독립 채점 안 함. 고정 규칙의 이득이 새 가중치에서도 유지됨. 2단계 판단은 비권장에서 보류로 바꿈. E·F도 재학습했는지 확인 필요 |

## Codex 상태

- 마지막 실제 확인: **2026-10-09 19:13 KST**
- 실험 ID: `accuracy81_HR_texture_pair_20261009`
- 상태: **완료 — 질감 입력 −0.111089%p, 확대 실험 미채택**
- 목적: 같은 HRNet 초기 가중치에서 원본 RGB 입력과 `[회색조, 저주파, 언샤프]` 입력의 추가 학습 효과 비교.
- 데이터: `/workspace/komap_model1/data_model382_seed1_fold1`; train70 내부 학습 52장 / 시편 분리 보류 18장. 공식 valid20·test10을 사용하지 않는다.
- 초기 가중치: `/workspace/komap_model1/outputs/model392_widthnorm/seed1/fold1/last.pt`
- 학습: 양 군 동일 4 epoch, seed 819349, patch512, scale2, 폭 정규화1230, FP32, batch1/누적8, LR3e-5, encoder LR 배율0.2, BN 통계 고정, 마지막 가중치 평가.
- 로컬 경로: `experiments/model349_compression/accuracy81_HR_texture_pair_20261009/`
- 서버: `root@168.188.128.36:20670`
- 서버 경로: `/workspace/model349_compress/accuracy81_HR_texture_pair_20261009/`
- GPU: **물리 GPU 3 사용 종료**. 실제 실험 프로세스가 모두 종료됐음을 확인했다. 새 작업 전 서버 점유를 다시 확인한다.
- 프로세스: controller 2309846, 최종 평가 및 finalizer 2310767 모두 종료. 다시 실행하지 않는다.
- 완료: 원본·질감 각각 4 epoch 학습, flip4 보류18 평가, 독립 채점 및 한국어 보고서 생성.
- 원본 학습: 정상 종료(0), 최종 loss 1.0664. 저장 체크포인트 SHA 및 BN 통계 고정 검증 완료.
- 최종 증거: `completion.json`, `independent_comparison.json`, `report.md`, `finalizer_completion.json`, `decision.json`. 독립 FP16 점수 원본 **77.691822%**, 질감 **77.580733%**. FP32 점수 원본 77.691884%, 질감 77.580957%. 공식 valid20 점수가 아니다.
- 판단: 개선 기준 실패. Al3Ni +0.037533%p, Primary Si −0.209596%p, Eutectic Si −0.229886%p. 같은 질감 설정으로 다른 fold를 확대 실행하지 않는다. 81% 목표는 미달성.

- 중복 확인: Claude의 `claude_grow_e_20261009`(E 구성원 추가, CPU 캐시 탐색)을 확인했으며 Codex는 이 탐색을 중복 실행하지 않는다.

### Codex 추가 학습 효과 확인 — 실행 전 등록

- 실험 ID: `accuracy81_HR_warmstart_audit_20261009`
- 상태: CPU 비교 완료. 기존 fold1 HRNet seed1 OOF 저장 확률과 방금 완료한 원본 입력 대조군을 같은 보류18 GT에서 비교한다.
- 차이: 질감 입력 효과와 별개로 추가 학습 자체의 효과를 확인한다. Claude의 B4 회색조 신규 구성원·E 평균 탐색과 겹치지 않는다.
- 경로: `experiments/model349_compression/accuracy81_HR_warmstart_audit_20261009/`, 서버 같은 ID.
- GPU: 미사용. 학습·추론 재실행 없음. 공식 valid/test 미사용.
- 결과: 기존 77.100650%, 추가 학습 77.691822%, **+0.591172%p**. 18장 저장 IoU 독립 재합산 일치. 추론 설정 메타데이터 일치 확인. 단일 fold 사후 진단이며 전체 성능 근거는 아직 없음.
- 다음 작업: 다른 fold의 같은 추가 학습 레시피 검증을 준비한다.
- 근거: `accuracy81_HR_warmstart_audit_20261009/result.json`, `report.md`, 서버 `inference_audit.json`.

### Codex fold2 추가 학습 — 실행 전 등록

- 실험 ID: `accuracy81_HR_warmstart_fold2_20261009` / 상태: 학습·보류18 평가·독립 비교 완료.
- 목적: fold1 원본 입력 추가 학습의 +0.591172%p 신호를 다른 fold에서 확인한다. 질감 채널은 사용하지 않는다.
- 레시피: fold1 대조군과 같은 seed819349, 4epoch, FP32 batch1/누적8, LR3e-5, encoder배율0.2, BN 통계 고정, 마지막 체크포인트.
- 데이터·부모: `data_model382_seed1_fold2`, `model392_widthnorm/seed1/fold2/last.pt`; train70 내부 분할만 사용. 공식 valid/test 사용 안 함.
- 서버·로컬: `/workspace/model349_compress/accuracy81_HR_warmstart_fold2_20261009/`, `experiments/model349_compression/accuracy81_HR_warmstart_fold2_20261009/`.
- GPU: 물리 GPU3 작업 종료. 새 작업 전 실제 점유 재확인.
- 순서: 원본 입력 추가 학습 → flip4 보류 평가 → 기존 fold2 OOF 캐시와 독립 비교. 중복 학습 금지.
- 완료 후 자동 비교·한국어 보고서 생성 작업을 연결했다. fold2 폭 정규화는 해당 학습 분할의 mode width **1278**(fold1은1230)로 부모 모델의 전처리를 유지한다.

- 결과: 기존 **77.979856%**, 추가 학습 **78.513970%**, **+0.534115%p**. 네 클래스 상승. 독립 저장 IoU 재합산 및 원본 SHA 보존 확인. 공식 valid20 점수가 아니다.
- 판단: fold1(+0.591172%p)에 이어 개선 신호. 같은 고정 레시피로 나머지 fold 확인을 진행할 근거가 생겼다.
- 근거: `accuracy81_HR_warmstart_fold2_20261009/independent_comparison.json`, `completion.json`, `report.md`, `decision.json`.

### Codex fold3 추가 학습 — 실행 전 등록

- 실험 ID: `accuracy81_HR_warmstart_fold3_20261009` / 상태: 완료 — 독립 채점 +0.207296%p.
- 목적: seed1 fold1 +0.591172%p, fold2 +0.534115%p 추가 학습 효과의 다른 fold 확인.
- 레시피: 기존과 같은 원본RGB, seed819349, 4epoch, FP32 batch1/누적8, LR3e-5/encoder배율0.2, BN통계 고정, 마지막 가중치.
- 데이터·부모: `data_model382_seed1_fold3`, `model392_widthnorm/seed1/fold3/last.pt`. train70 내부 학습/보류만 사용; 공식 valid/test 미사용.
- 경로: `/workspace/model349_compress/accuracy81_HR_warmstart_fold3_20261009/`; 로컬 같은 ID.
- GPU: 물리GPU3 사용 종료. 다른 작업 전 실제 점유 재확인.
- 순서: 추가 학습 → flip4 보류 평가 → 기존 fold3 OOF와 독립 비교. 기존 fold1/2는 재실행하지 않는다.

- 결과: 보류17 기존 **82.763613%**, 추가 학습 **82.970908%**. Eutectic Si +0.736278%p, Primary Si +0.140208%p, Al −0.027680%p, Al3Ni −0.019623%p. 추론 메타데이터 일치·원본 SHA 보존·독립 IoU 재합산 확인.
- 다음: 같은 레시피 fold4 확인 후 전체70장 및 E 결합 효과 평가. 이 점수는 공식 valid20 또는 81% 달성 근거가 아니다.
- 근거: `accuracy81_HR_warmstart_fold3_20261009/independent_comparison.json`, `report.md`, `decision.json`.

### Codex fold4 추가 학습 — 실행 전 등록

- 실험 ID: `accuracy81_HR_warmstart_fold4_20261009` / 상태: 학습·평가·독립 채점 완료. 보류17 기존 78.234944%, 추가 학습 78.888046%, +0.653102%p. 추론 메타데이터 일치. GPU 사용 종료.
- 목적: seed1 원본RGB 추가 학습 효과의 마지막 fold 확인. fold1 +0.591172, fold2 +0.534115, fold3 +0.207296%p.
- 레시피: seed819349, 4epoch, FP32 batch1/누적8, LR3e-5/encoder배율0.2, BN통계 고정, 마지막 가중치. 후보/epoch 추가 선택 없음.
- 데이터·부모: `data_model382_seed1_fold4`, `model392_widthnorm/seed1/fold4/last.pt`; train70 내부 분할만 사용. 공식 valid/test 미사용.
- 경로: `/workspace/model349_compress/accuracy81_HR_warmstart_fold4_20261009/`; 로컬 같은 ID.
- GPU: 물리GPU3 유휴 확인. Claude B4 회색조(GPU4/6/7)와 중복 없음.
- 다음: flip4 보류17 독립 채점 → 70장 통합 및 고정 E의 HRseed1 교체 효과 확인. 앞선 fold는 재학습하지 않는다.

### Codex HR 추가 학습의 E 영향 — 준비 등록

- ID: `accuracy81_HR_warmstart_E_oof_20261009` / 상태: CPU 채점 완료(exit0). E 79.985862 → 80.060908%, +0.075046%p; 네 fold 모두 개선. HR 단독 +0.498313%p.
- 데이터: train70 저장된 시편 분리 예측만 사용. E 비율0.5B4+0.25HRseed1+0.25HRseed2를 고정하고 HRseed1만 추가 학습 예측으로 교체.
- 평가: 동일70장 원본 E 점수79.9858620081% 재현, 전체·fold별 변화 및 원본HR/추가학습HR 비교. GPU·공식 valid/test 사용 없음.
- 한계: fold1 결과를 본 뒤 확대된 레시피이므로 선택 편향 없는 최종 OOF 검증으로 주장하지 않는다.
- Claude E 신규 구성원 추가·B4회색조와 차이: 기존 HRseed1의 고정 추가 학습 효과만 평가. 결합 규칙/비율 탐색 안 함.

## Codex 최근 완료 작업 — 같은 조건의 반복 방지

모든 경로는 `experiments/model349_compression/` 아래에 있다. 최종 결과는 각 디렉터리의 JSON·보고서를 확인한다.

| 실험 ID | 내용 | 판단 |
|---|---|---|
| `accuracy81_stage1_fresh_20261009` | 고정된 R + E 마스크 규칙 전체 fresh 추론, valid20·test10 수집 | 완료. valid20 독립 점수 80.474424%. 재학습 검증은 아님. |
| `accuracy81_E_oof_bias_20261009` | train70 E 예측의 클래스 log-odds 보정 | 손해, 미채택 |
| `accuracy81_E_oof_smoothing_20261009` | E 확률 Gaussian 평활화 | 이득 없음, 미채택 |
| `accuracy81_E_oof_powerpool_20261009` | E 구성원의 일반화 평균 결합 | 손해, 미채택 |
| `accuracy81_E_error_diagnostic_20261009` | train70 E 오류 및 구성원 공통 오류 진단 | 진단 완료 |
| `accuracy81_E_context_eut_20261009` | Eutectic Si 문맥 보정기, 4-fold × 400 updates | −0.022152%p, 미채택 |
| `accuracy81_E_pair_iou_20261009` | Al/Eut 전용 IoU 손실 보정기, 4-fold × 400 updates | −0.048615%p, 미채택 |

위 train70 메타 보정 실험은 기저 모델까지 재학습하는 엄밀한 nested 평가가 아니다. Claude의 OOF90 설계와 동일한 평가로 해석하지 않는다.

## 새 작업 등록 양식

- 담당자 / 갱신 시각:
- 실험 ID / 상태(준비·실행·평가·완료·실패):
- 가설과 기존 작업 대비 차이:
- 데이터 / 초기 가중치 / 고정할 선택 규칙:
- 로컬·서버 경로 / GPU / 실제 PID:
- 현재 단계 / 다음 작업:
- 결과 경로 / 점수 / 한계 / 채택 판단:

## Codex 전체 실험·검증 목록

기록 범위: 이 세션에서 사용한 `experiments/model349_compression/`의 모든 실험 디렉터리와 `results/`의 비교·채점 기록. 2026-10-09 파일 근거로 정리했다. 기존 upstream 모델 학습을 Codex가 새로 수행한 것으로 표시하지 않는다. 보고서의 과거 결정은 당시 기준이며, 현재 대표 후보·상태는 위 최신 상태가 우선한다.

점수는 기록된 평가 범위를 따른다. 캐시 재생, 공식 valid20 재추론, train70 메타 OOF, 보류 fold 파일럿, 속도·저장량 측정을 서로 같은 성능 검증으로 해석하지 않는다. 결과 파일이 없는 항목은 완료로 추정하지 않는다.

### accuracy81_20261008

- 작업 폴더: [accuracy81_20261008](experiments/model349_compression/accuracy81_20261008/)
- [B4 D4 원본 valid20 재추론 결과](experiments/model349_compression/accuracy81_20261008/b4d4_report.md): 20개 PNG와 정답을 독립 채점했고 실행 metrics의 이미지별 80개 IoU와 일치했다. 이미지별 4-class IoU 평균을 20장 평균하며 빈 union은 0이다. 원본 14개 가중치와 기존 계수·라우팅·고해상도 TENT5는 유지했다. B4 322/326/329만 실제 8방향 TTA를 활성화했다. 평균 +0.0144776%p, 최악 A74_step3_sample4 -0.2043264%p. 개선 13장 / 악화 7장. 조건부 시편 그룹 bootstrap 95% 구간 [-0.0290002, +0.0633737]%p (20,000회, seed81795, 19그룹). 반복 사용한 valid의 사후 진단으로 선택 편향을 제거하지 않으며 독립 일반화 증거가 아니다. 전체 원본 재추론 단회 시간은 897.05초, GPU7이다. HR 추가 TTA 없는 B4 D4 실험이며 속도 개선은 없다. 환경 변동이 있는 다른 실행과 정밀 속도 비교하지 않는다. 아직 81%에 못 미치며 결합 TTA/대조 학습 결과를 기다린다. 기존 base14/원본 가중치/릴리스는 변경하지 않는다. test 점수는 없다.
- [B4 D4 + HRNet flip4 결합 TTA 결과](experiments/model349_compression/accuracy81_20261008/combined_tta_report.md): 모든 원본 valid20 입력에서14개원본가중치를사용해추론했다. B4 322/326/329 D4, HR336/347 flip4를활성화하고 다른계수/라우팅/native269/고해상도TENT5/후처리는유지했다. 20개 PNG/GT 독립재채점은서버metrics의80 IoU와정확히일치한다. 빈union=0인이미지별4클래스IoU평균후20장평균이다. 최악 A186_step2_sample2 -0.5006887%p. 개선15장/악화5장. 조건부19그룹 bootstrap95% [-0.0493973, +0.0843454]%p (20,000회 seed81795). 반복valid개발결과의사후진단이고 후보선택편향을제거하지않았으므로 독립일반화증거가아니다. 가중치14개/1,755,428,956bytes 그대로다. 전체 단회시간 1350.30초(GPU7). HR-only 중단재개구간457.99초와혼동하지않는다. 신규81%목표는미달이다. rawvalid최고관측후보로기록하지만 작은평균상승과상당한국소손실/시간비용이있어현재base는교체하지않는다. 학습및배율후보와비교를이어간다. test점수없음.
- [전역 융합48개 탐색 결과](experiments/model349_compression/accuracy81_20261008/fusion_report.md): 기존14개fold/라우팅/후처리는 유지하고 모든영상에 동일한 Al3Ni326·329/Eutectic336 융합계수를 적용했다. 저장된identity확률재생이며 원본영상 재추론 성적이 아니다. 선정계수는326Al3Ni0.5,329Al3Ni0.5,336Eutectic0.75. 최악A71_step2_sample1은−0.2878223%p였다. 조건부specimen-bootstrap95%구간 [−0.0382387,+0.0359856]%p로0포함; 고정grid에서specimen제외선택은평균−0.1229598%p였다. 이진단은이미반복사용한valid와선택된base에조건부이며 독립일반화평가가아니다. 개선폭이매우작고국소손실/선택불안정성이있어채택하지않고원본재추론은추가실행하지않는다. 결과/YAML을탐색기록으로동결하고GPU의TTA실험을우선한다. 기존base는유지한다.
- [영상 경계 guided logit 후처리 탐색 결과](experiments/model349_compression/accuracy81_20261008/guided_report.md): 기준20마스크/80 IoU가 기존best14 replay와정확히같고 최고재계산도일치했다. 공식지표는이미지별4classIoU평균후20장평균,빈union=0이다. 저장확률의반복valid탐색이며전체fresh원본재추론/독립일반화성적이아니다. 후보수 28, 최고 80.2309769%. 최고는기준그대로이며새개선없음. 합성 radius0identity/constantfield/constantguide/finite 검사통과. 원본영상20개SHA는canonical원본manifest와같았다. 구현수식은[원논문](https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf)의gray local linear filter이며border는REFLECT_101로고정했다. 변경후보를채택하거나추가GPU원본추론하지않는다. 최고차이0/구간0은동일모델을자신과비교한것이며일반화보장이아니다. source/protocol/입력cache/SHA를결과에보존했고원본모델·릴리스는변경하지않았다. test점수없음.
- [고해상도 soft ensemble 탐색 결과](experiments/model349_compression/accuracy81_20261008/high_ensemble_report.md): 원본14의 다른분기/계수/후처리를 유지하고 4개 또는5개 고해상도 경로의 산술·기하평균을 고정 비율로 섞었다. 전체9개 후보(기준포함)를 결과 전 고정했으며 기존cached base의80 IoU와최고재실행이정확히일치했다. 기준하드라우팅 자체가 최고다. 8개 변경후보는 모두악화되어채택/새GPU추론하지않는다. 저장확률의조합성적이며 전체원본재추론성적/독립일반화평가가아니다. 최고가기준그대로여서선정후보차이0과조건부구간0은단지동일모델비교이며일반화보장으로해석할수없다. test점수없음.
- [HRNet336 flip4 부분 결과 진단](experiments/model349_compression/accuracy81_20261008/hr336_partial_report.md): GPU7 타사용자점유를감지해부모TTA작업이347시작전종료했다. 완료322/326/329/base/336산출물은보존했고,현재재개controller PID2146242는GPU7유휴를기다린다. 기타GPU/타사용자PID에신호하지않았다. 완료336확률20개를Mac으로수집해전부서버SHA와대조했다. 이확률만이전identity캐시에치환한탐색진단이며전체HR336+347flip4원본재추론결과가아니다. 기존캐시baseline20개IoU정확재현을확인했다. 평균+0.0061138%p지만최악A181_step1_sample1−0.4351261%p다. 일반화/공식전체성능개선주장이나채택하지않는다. GPU가비면347/269만재개하여완료5분기를중복하지않고전체마스크를생성한다. 이후기존TTA프로토콜의B4/결합후보를순차검증한다.
- [기존 HR392의 앙상블 보완성 탐색](experiments/model349_compression/accuracy81_20261008/hr392_complement_report.md): Model401이 원본 valid20에서 생성한 두 seed × 각4fold 평균 확률 40개를 재사용했다. 모든 파일을 서버 manifest SHA와 로컬 SHA로 대조했다. team→main 클래스 순서를 맞추고 native 크기/유한 값/합 정규화를 확인했다. 기준 캐시의 모든 80개 IoU는 기존 replay와 정확히 같고 최고 후보는 원래 전체 predict 경로로 다시 채점해 정확히 일치했다. 최고 평균 변화 +0.0070461%p. 최악 A186_step2_sample2 -0.1324724%p. 조건부 그룹 bootstrap95% [-0.0259319, +0.0407002]%p. 이 7개 고정 후보 범위의 시편 제외 선택 진단 평균 -0.0197856%p. 독립 일반화 증거가 아니다. 최고 후보는 기존14개에 HR392 가중치8개를 더 쓰는 22개 구성이며 저장·추론 비용 증가가 예상되지만 실제 시간을 측정하지 않았다. 소폭 평균 상승, Eutectic 평균 악화와 시편 제외 선택 불안정성으로 채택하지 않고 새 GPU 원본 추론도 예약하지 않는다. 기존 base와 릴리스는 유지한다. 이것은 재사용 valid 저장 확률의 탐색 성적이며 전체 새 후보를 원본 영상부터 재추론한 확정 성적이 아니다. 기존 Model401 valid 결과를 읽은 뒤 설계했으므로 사전 고정된 7개 후보라도 탐색 전체가 독립 확증이 되지는 않는다. test mIoU 없음.
- [HRNet flip4 완료 결과와 이미지별 변화](experiments/model349_compression/accuracy81_20261008/hrflip4_report.md): 원본 valid20 재추론의 완료 분기는 중단 후 재개 때 보존했다. 마스크/GT 독립 재채점은 실행 metrics와 모두 일치한다. 이미지별 4-class IoU 평균(빈 union=0)을 다시 20장 평균한 점수다. 같은 valid로 개발한 결과이며 독립 일반화 증거가 아니다. 평균 +0.0088082%p, 최악 A181_step1_sample1 -0.4375342%p. 개선 14장 / 악화 6장. 조건부 그룹 bootstrap 95% 구간 [-0.060742119250181865, 0.07101554244480614] %p. 이 구간은 후보 선택 편향을 제거하지 않은 사후 진단이다. 채택하지 않고 기존 base14를 유지한다. 14개 원본 가중치/코드/설정 SHA가 보존됐고 완료5분기 확률도 재개 전후 같았다. 시간457.99초는 재개 구간만이며 전체 단회 추론시간이나 속도 개선 근거가 아니다. test 점수는 없다.
- [Al3Ni·Primary Si 상대 bias 탐색 결과](experiments/model349_compression/accuracy81_20261008/phase_bias_report.md): 기준20마스크/80 IoU가 기존best14 replay와정확히같고 최고재계산도일치했다. 공식지표는이미지별4classIoU평균후20장평균,빈union=0이다. 저장확률의반복valid탐색이며전체fresh원본재추론/독립일반화성적이아니다. 후보수 49, 최고 80.2309769%. 최고는기준그대로이며새개선없음. 고정49후보의시편제외선택진단은-0.0819895%p. 같은valid에조건부인선택민감도이며새시편평가가아니다. 변경후보를채택하거나추가GPU원본추론하지않는다. 최고차이0/구간0은동일모델을자신과비교한것이며일반화보장이아니다. source/protocol/입력cache/SHA를결과에보존했고원본모델·릴리스는변경하지않았다. test점수없음.
- [81% 목표: 전역 보정 탐색 결과](experiments/model349_compression/accuracy81_20261008/report.md): 전역 보정120후보의 최고는 기존 base14 설정이었다. 저장 확률 재생 최고80.2309769%; 원본영상 재추론 점수80.2312313%와 구분한다. 이번 단계에서 정확도 상승은 없었다. base14원래설정의20마스크/80개클래스IoU가 이전 재생과 정확히 같았다. 선정후보를 원래 predict 함수로 재계산해120grid빠른계산 결과와 정확히 일치했다. 고정grid에서 specimen을 하나씩 제외해 선택한 결과는 base14대비평균−0.2211463%p였다. 이는 보정계수 선택이 작은 valid에 불안정함을 보여주는 진단이며, 이미valid로고른base14/고정기존가중치에 조건부이므로 독립 일반화평가가 아니다. 다음단계는 hrflip4→b4d4→combined_tta 원본영상 재추론이다. 계수/fold선택을 추가변경하지 않으며 타사용자작업/원본가중치를보존한다. 81%목표는 미달성이다.
- [전역 공간 후처리100개 탐색](experiments/model349_compression/accuracy81_20261008/spatial_report.md): 기존baseline20마스크/IoU정확재현 및최고후보원래predict함수20IoU정확일치확인. 저장확률재생이며원본전체재추론성적/독립일반화증거가아니다. PrimarySi gaussian_blend0.5/cleanup128은유지된다. 평균+0.0044582%p,최악A71−0.0819971%p. 조건부group bootstrap95%[−0.0150950,+0.0253183]%p,고정 grid시편제외선택−0.0225282%p. 매우작은상승이므로채택/추가raw실행하지않고탐색YAML만보존한다. 같은valid반복선택의한계를유지한다.
- [정확도용 TTA 원본 재추론 검증](experiments/model349_compression/accuracy81_20261008/tta_report.md): 빈 union=0인 이미지별4클래스평균 후20장평균. PNG/GT 독립재채점과실행metrics일치 확인. 동일valid반복개발평가이며 독립일반화 증거가 아니다. 가중치14개/1,755,428,956바이트 유지. 시간은 속도개선 근거가 아니다. 중단 후 재개한 경우 전체 소요시간을 측정하지 못했고 재개 구간만 표시한다. 기존릴리스/원본가중치보존. test mIoU 없음.
- [b4d4_review.json](experiments/model349_compression/accuracy81_20261008/b4d4_review.json): status=exploratory_candidate_pending_combined_and_learning; base_unchanged=True; raw_valid20_independent_scoring=True; miou_percent=80.24570888454825; delta_pp=0.014477564744441729; worst_image=A74_step3_sample4; worst_image_delta_pp=-0.20432637920910768; bootstrap_groups=19; independent_generalization_verified=False; test_score_available=False; elapsed_seconds_single_run=897.0472368597984
- [combined_tta_review.json](experiments/model349_compression/accuracy81_20261008/combined_tta_review.json): status=accuracy_candidate_recorded_not_promoted; miou_percent=80.25488413100351; delta_pp=0.02365281119970364; worst_image=A186_step2_sample2; worst_image_delta_pp=-0.5006886944204814; elapsed_seconds_single_run=1350.29889345821; independent_png_scoring_agrees=True; independent_generalization_verified=False; base_unchanged=True; test_score_available=False
- [failure_gpu_occupied_initial.json](experiments/model349_compression/accuracy81_20261008/failure_gpu_occupied_initial.json): time=1791460846.2608564
- [fusion_decision.json](experiments/model349_compression/accuracy81_20261008/fusion_decision.json): selected_yaml=fusion_selected_exploratory.yaml; selected_yaml_sha256=e3764804ade0eeb516aee96abda6341c0dae3215c018c009608be93c85a2fabc; adopted=False; raw_reinference_launched=False; reason=Cached gain only +0.001920823pp with worst loss -0.287822337pp and conditional CI spanning zero. Keep current base; prioritize running TTA experiments.; selection_bias=Same adaptively reused valid20; no independent generalization confirmation.
- [guided_decision.json](experiments/model349_compression/accuracy81_20261008/guided_decision.json): adopted=False; base_unchanged=True; raw_inference_scheduled=False; reason=baseline best; all changed candidates lower; independent_generalization_verified=False
- [high_ensemble_decision.json](experiments/model349_compression/accuracy81_20261008/high_ensemble_decision.json): adopted=False; raw_gpu_inference_scheduled=False; base_unchanged=True; reason=all 8 changed candidates below existing hard routing; independent_generalization_verified=False
- [hr392_complement_decision.json](experiments/model349_compression/accuracy81_20261008/hr392_complement_decision.json): adopted=False; raw_gpu_inference_scheduled=False; base_unchanged=True; reason=tiny mean improvement with additional 8 checkpoints; Eutectic decline; negative fixed-grid LOSO diagnostic; mean_delta_pp=0.00704612422223215; independent_generalization_verified=False
- [hrflip4_decision.json](experiments/model349_compression/accuracy81_20261008/hrflip4_decision.json): scope=post-result descriptive paired group-bootstrap; conditional on reused valid and fixed original selected base, not independent generalization; seed=817; draws=20000; delta_pp=0.008808212100855917; improved_images=14; worsened_images=6; adopted_as_base=False; reason=Gain only +0.008808pp with worst loss -0.437534pp; continue preregistered B4/combined and paired training without replacing current base.
- [phase_bias_decision.json](experiments/model349_compression/accuracy81_20261008/phase_bias_decision.json): adopted=False; base_unchanged=True; raw_inference_scheduled=False; reason=baseline best; all changed candidates lower; independent_generalization_verified=False

### accuracy81_E_context_eut_20261009

- 작업 폴더: [accuracy81_E_context_eut_20261009](experiments/model349_compression/accuracy81_E_context_eut_20261009/)
- [Eutectic Si 31×31 문맥 보정 학습 실험](experiments/model349_compression/accuracy81_E_context_eut_20261009/report.md): 7,713개 파라미터의 팽창 합성곱 보정기로 기존 E의 Eutectic Si 점수만 ±1 이내에서 수정했다. train70에서 grouped fold 하나씩을 제외하고 나머지 영상으로 400회씩 학습했다. 고정 마지막 체크포인트를 평가했으며 valid/test는 읽지 않았다. 기준 **79.9858620081%**, 후보 meta-OOF **79.9637103823%**, 차이 **-0.0221516258%p**로 손해다. 70장 클래스 IoU를 로컬에서 독립 재합산해 확인했다. 후보는 기각하고 valid 후속은 실행하지 않는다. 이는 기존 base OOF를 재사용한 탐색용 meta-OOF이며, base까지 중첩 재학습한 독립 일반화 검증이 아니다. 기준 모델과 학습 특징은 불변이다. 초기 결정적 CUDA CE 오류는 수식이 같은 log-softmax CE로 수정했다. NVIDIA PID와 컨테이너 PID가 달라 생긴 감시 종료는 단일 CUDA 생성 전후 PID 등록으로 해결했다. 4개 fold 학습 후 결과 JSON의 NumPy bool 저장 오류는 체크포인트 평가만 다시 실행해 복구했다. 복구 전후 네 체크포인트 SHA가 같아 재학습하지 않았음을 확인했다. 실패 시도와 로그는 서버의 attempt 폴더에 보존했다.
- [completion.json](experiments/model349_compression/accuracy81_E_context_eut_20261009/completion.json): finished=True; folds=4; updates=1600; screen_pass=False
- [decision.json](experiments/model349_compression/accuracy81_E_context_eut_20261009/decision.json): decision=reject; delta_pp=-0.02215162581423158; screen_pass=False; independent_saved_iou_recalculation=True; valid_followup=False; base_changed=False; recovery_did_not_retrain=True; checkpoints_unchanged_on_recovery=True

### accuracy81_E_error_diagnostic_20261009

- 작업 폴더: [accuracy81_E_error_diagnostic_20261009](experiments/model349_compression/accuracy81_E_error_diagnostic_20261009/)
- [E train70 OOF 오류 진단](experiments/model349_compression/accuracy81_E_error_diagnostic_20261009/report.md): 정답을 이용한 오류 분석이며 제출 가능한 모델이나 성능 점수가 아니다. valid/test·GPU는 사용하지 않았다. 70장 저장 집계를 독립 재합산해 일치했다. Eutectic Si의 오류 중 약 71%는 구성원들이 같은 오답을 내고 있다. 평균 방식 변경만으로 해결하기 어려운 공통 오류가 많으므로 다음 후보는 Eutectic Si의 학습 표현·손실을 바꾸는 구성원이 필요하다. 약 23%는 개별 구성원이 맞혔으나 평균에서 잃은 픽셀로, 결합 개선 여지도 존재한다. 이는 정답을 본 사후 진단이며 실제 교정 가능 비율을 뜻하지 않는다.
- [completion.json](experiments/model349_compression/accuracy81_E_error_diagnostic_20261009/completion.json): images=70; finished=True

### accuracy81_E_oof_bias_20261009

- 작업 폴더: [accuracy81_E_oof_bias_20261009](experiments/model349_compression/accuracy81_E_oof_bias_20261009/)
- [E 클래스 편향 train70 OOF 검사](experiments/model349_compression/accuracy81_E_oof_bias_20261009/report.md): B4 + HRNet 2시드의 고정 E에 Al3Ni·Al·Eutectic Si log-odds 편향 각각 −0.1 / 0 / +0.1을 적용한 27개 설정을 비교했다. 네 grouped fold마다 나머지 세 fold에서 보정 값을 고르고 제외한 fold에서 평가했다. 기존 E OOF 79.9858620%를 정확히 재현했다. 보정의 fold 제외 값은 79.9824486%, 기준 대비 −0.0034134%p다. 사전 +0.10%p / 3개 fold 개선 / 최악 fold −0.10%p 기준을 통과하지 못해 기각했다. 전체 70장에 맞춘 선택 값은 검증값으로 사용하지 않는다. 저장된 70장 클래스 IoU에서 점수를 재계산해 일치했다. train70 GT만 사용했고 valid/test GT 및 GPU는 읽거나 사용하지 않았다. 부모 확률 SHA는 불변이며, 진행 중인 1단계 원본 영상 재추론 설정과 기존 base는 바꾸지 않았다.
- [completion.json](experiments/model349_compression/accuracy81_E_oof_bias_20261009/completion.json): images=70; screen_pass=False; valid_read=False; test_read=False; gpu_used=False
- [decision.json](experiments/model349_compression/accuracy81_E_oof_bias_20261009/decision.json): decision=reject_no_valid_or_gpu_followup; baseline_train_oof_percent=79.98586200807742; nested_fold_train_oof_percent=79.98244858976142; delta_pp=-0.003413418316000616; recomputed_from_all_70_saved_class_ious=True; source_sha_exact=True; not_official_valid_score=True; existing_stage1_recipe_changed=False

### accuracy81_E_oof_powerpool_20261009

- 작업 폴더: [accuracy81_E_oof_powerpool_20261009](experiments/model349_compression/accuracy81_E_oof_powerpool_20261009/)
- [E 확률 평균 방식 train70 OOF 실험](experiments/model349_compression/accuracy81_E_oof_powerpool_20261009/report.md): 동일 구성원·가중치 0.5/0.25/0.25에서 산술·기하·제곱근·제곱·조화 평균을 비교했다. grouped fold를 제외하고 나머지 3개에서 평균 방식을 골랐다. 기준 79.9858620081%, fold 제외 79.9838121297%, 차이 -0.0020498783%p다. 전체 train70에 맞춘 최선은 power=2이나 fold 제외 결과는 손해여서 기각한다. 저장 클래스 IoU로 기준·fold 제외 점수를 독립 재계산해 일치했다. valid·test·GPU는 사용하지 않았고 부모 SHA는 불변이다. valid 후속 실행은 하지 않는다.
- [completion.json](experiments/model349_compression/accuracy81_E_oof_powerpool_20261009/completion.json): images=70; screen_pass=False; valid_read=False; test_read=False; gpu_used=False
- [decision.json](experiments/model349_compression/accuracy81_E_oof_powerpool_20261009/decision.json): decision=reject; nested_delta_pp=-0.0020498783344322646; reason=Nested fold score decreases and preregistered gain gate fails; independent_saved_iou_recalculation=True; valid_followup=False; base_changed=False

### accuracy81_E_oof_smoothing_20261009

- 작업 폴더: [accuracy81_E_oof_smoothing_20261009](experiments/model349_compression/accuracy81_E_oof_smoothing_20261009/)
- [E 국소 확률 평활화 train70 OOF 실험](experiments/model349_compression/accuracy81_E_oof_smoothing_20261009/report.md): 원본과 Gaussian sigma 1·2 픽셀 × 혼합 0.1·0.25·0.5의 7개 설정을 비교했다. 시편 grouped fold를 하나씩 제외하고 나머지 3개 fold에서 설정을 선택했다. 기준과 fold 제외 결과는 모두 **79.9858620081%**로 변화가 없다. 전체 train70에서도 평활화하지 않은 원본이 선택됐다. 저장된 70장 클래스 IoU를 별도로 재평균해 확인했다. 개선 gate를 통과하지 못해 기각하고 valid 후속 실험을 하지 않는다. train70 OOF만 읽었으며 GPU·valid·test는 사용하지 않았다. 부모 SHA 불변 및 exit0 확인.
- [completion.json](experiments/model349_compression/accuracy81_E_oof_smoothing_20261009/completion.json): images=70; screen_pass=False; valid_read=False; test_read=False; gpu_used=False
- [decision.json](experiments/model349_compression/accuracy81_E_oof_smoothing_20261009/decision.json): decision=reject; reason=Identity selected in full train and nested folds; no positive gain; baseline_percent=79.98586200807742; nested_percent=79.98586200807742; delta_pp=0.0; independent_saved_iou_recalculation=True; valid_followup=False; base_changed=False

### accuracy81_E_pair_iou_20261009

- 작업 폴더: [accuracy81_E_pair_iou_20261009](experiments/model349_compression/accuracy81_E_pair_iou_20261009/)
- [Al·Eutectic 전용 IoU 보정 학습 실험](experiments/model349_compression/accuracy81_E_pair_iou_20261009/report.md): 같은 31×31 문맥 보정기와 4fold×400 업데이트를 사용하되, 원본 Al·Eutectic 예측 픽셀만 두 클래스 사이에서 바꾸도록 제한했다. 손실은 0.2 pair BCE + 0.8 crop별 soft Eutectic IoU + 잔차 L2다. Primary Si·Al3Ni 예측 마스크 불변을 70장마다 검사했다. 기준 79.9858620081%, 후보 meta-OOF 79.9372465929%, 차이 **-0.0486154152%p**로 손해다. 클래스별 변화(route 순서)는 [0.0, 0.0, -0.028632355777738722, -0.1658293048352078] %p다. 70장 저장 IoU를 독립 재합산해 점수와 rare class 불변을 확인했다. 후보는 기각하고 valid 후속은 실행하지 않는다. train70만 사용하고 valid/test는 읽지 않았다. 기존 base OOF를 재사용한 meta-OOF이므로 엄격한 중첩 base 학습 검증이 아니다. 부모 특징 SHA는 불변이며 원본 base를 유지한다.
- [completion.json](experiments/model349_compression/accuracy81_E_pair_iou_20261009/completion.json): finished=True; folds=4; updates=1600; screen_pass=False
- [decision.json](experiments/model349_compression/accuracy81_E_pair_iou_20261009/decision.json): decision=reject; delta_pp=-0.048615415153236086; rare_class_prediction_invariance_checked=True; independent_saved_iou_recalculation=True; valid_followup=False; base_changed=False

### accuracy81_HR_texture_pair_20261009

- 작업 폴더: [accuracy81_HR_texture_pair_20261009](experiments/model349_compression/accuracy81_HR_texture_pair_20261009/)
- 완료·미채택: 보류18 독립 점수 −0.111089%p. [보고서](experiments/model349_compression/accuracy81_HR_texture_pair_20261009/report.md), [결정](experiments/model349_compression/accuracy81_HR_texture_pair_20261009/decision.json). 원본·데이터 SHA 불변 확인.
- [HRNet 질감 입력 대조 실험](experiments/model349_compression/accuracy81_HR_texture_pair_20261009/README.md): 같은 HRNet seed1/fold1 체크포인트에서 원본 RGB 입력과 [회색조, 저주파, 언샤프] 입력을 같은 예산으로 추가 학습한다. train70의 fold1 분할(학습 52장, 보류 18장)만 사용한다. 공식 valid20, test10, Claude의 OOF90 작업과는 별도다. 각 입력을 4 epoch 학습한 마지막 체크포인트로 flip4 추론한다. 후보의 보류 점수가 대조군보다 0.1%p 이상 높고 Al3Ni·Primary Si 각각의 하락이 0.1%p 이하일 때만 다른 fold 검증을 진행한다. 단일 fold 결과로 전체 성능 또는 81% 달성을 주장하지 않는다. 서버의 run_sequence.py가 원본 학습 → 질감 학습 → 원본 평가 → 질감 평가를 순차 실행한다. started.json이 있으면 중복 실행을 거부한다. GPU 3의 잠금 및 NVML PID를 확인하며 다른 프로세스가 나타나면 우리 자식만 중단한다. 원본 체크포인트와 데이터 SHA를 종료 시 재확인한다. 완료 후 verify_results.py로 저장된 FP16 확률을 정확한 RGB GT 팔레트에 독립 채점한다. FP32 평가기의 점수와 FP16 독립 점수를 구분한다. 두 점수는 양자화 때문에 소폭 다를 수 있다.
- [prelaunch_split_path_failure.json](experiments/model349_compression/accuracy81_HR_texture_pair_20261009/prelaunch_split_path_failure.json): 

### accuracy81_b4all70_20261009

- 작업 폴더: [accuracy81_b4all70_20261009](experiments/model349_compression/accuracy81_b4all70_20261009/)
- [B4 세 fold의 train70 추가 학습 검증](experiments/model349_compression/accuracy81_b4all70_20261009/report.md): 이미지별4클래스IoU 평균 후valid20평균,emptyunion0. 독립PNG/GT채점과모든80IoU 일치 확인. 같은valid 반복개발평가이며독립일반화증거가아니다. 원본 base 80.2312313%, 새 후보 80.1189330%, 변화 -0.1122983%p. 14가중치 1,755,301,831bytes. 전체freshvalid 단회 499.38초. 속도개선으로판정하지않는다. 기존3fold를모두유지했고새파라미터수/구조/FP32dtype변경없음. 학습후세모델은train70을보므로OOF모델이아니다. 최종checkpoint의키/shape/dtype/finite와BN buffer exact를원본과대조했고모델tensor실제변경/3fold동일800sample순서를확인했다. 원본source/14weights 보존. test정답및test mIoU없음. 비교는전체추가학습효과다. train70의노출만분리한동일계산량대조군은없다. 최초random3채널합성입력의비유한gradient와자연전처리train입력게이트수정근거를실패snapshot/수정문서에보존했다. loss scale/비유한값판정은완화하지않았다. 세 fold는 각각 400업데이트를 마쳤고 학습 이미지 70장을 모두 보았다. 학습 시간은 fold1 607.48초, fold3 599.30초, fold4 751.09초였다. 세 fold의 800개 샘플 순서는 동일했다. 서버 completion과 추론 completion이 모두 정상 종료를 기록했고, 수집 시 B4 controller와 
- [decision.json](experiments/model349_compression/accuracy81_b4all70_20261009/decision.json): candidate=B4 Model329 folds 1, 3, 4: 400 train70 continuation updates each; decision=reject; reason=Independent valid20 image mIoU decreased against canonical base and same-GPU3 control; all four mean class IoUs decreased.; candidate_miou_percent=80.1189329934781; canonical_base_miou_percent=80.23123131980381; delta_vs_canonical_pp=-0.11229832632571046; same_gpu3_control_miou_percent=80.23122801008668; delta_vs_same_gpu3_control_pp=-0.11229501660858787; worst_image=A71_step2_sample1; worst_image_delta_pp=-2.424426781799649; independent_png_gt_20_images_80_ious_verified=True; server_source_51_files_sha256_verified=True; server_parent_14_checkpoints_sha256_verified=True; original_base_retained=True; release_pointer_unchanged=True; target_81_percent_met=False; independent_generalization_verified=False; test_used=False; report=report.md; comparison=comparison.json; archive_sha256=8994e0ce6c8ad482d0ac86f91313b49a23bab7e5aea8196f7fec23fbb5a3c
- [training_completion_snapshot.json](experiments/model349_compression/accuracy81_b4all70_20261009/training_completion_snapshot.json): 

### accuracy81_candidate_oracle_20261009

- 작업 폴더: [accuracy81_candidate_oracle_20261009](experiments/model349_compression/accuracy81_candidate_oracle_20261009/)
- [R + E 1단계 마스크 결합 후보](experiments/model349_compression/accuracy81_candidate_oracle_20261009/report.md): 대표 후보는 **1단계 R + E**다. R은 Model349_exh14, E는 P100 Model329 B4 4-fold 및 Model392 HRNet 2-seed × 4-fold 앙상블이다. E가 Al3Ni 또는 Al로 예측한 픽셀만 E를 따르고 나머지는 R을 유지한다. 보정기 F는 필요 없다. 2단계 F는 전체 valid 탐색에서는 +0.0261%p지만 조건부 시편 제외 선택에서 1단계보다 낮아 대표 후보로 채택하지 않는다. 릴리스 대비 +0.0848%p라는 사실만으로 추가 단계의 유효성을 주장하지 않는다. 확신도 제한 80.5123%도 같은 반복 valid 탐색의 별도 결과이며 대표 후보로 승격하지 않는다. `fusion_screen.py`는 같은 시편 접두사(A63 등)의 모든 영상을 선택 자료에서 제외한다. 남은 영상에서 **69개 실행 × 각 30개 규칙 전체**를 다시 고르고, 제외한 영상의 IoU를 기록한다. 실행 E를 전체 valid에서 먼저 고정하고 규칙 30개만 재선택한 값이 아니다. 공식 지표와 같이 제외 영상들의 image mIoU를 20장 평균한다. 독립 재계산 및 선택 내역 대조는 `fusion_screen_cv_audit_result.json`에 기록한다.
- [완료 후보 상보성·클래스 마스크 융합 검사](experiments/model349_compression/accuracy81_candidate_oracle_20261009/report_before_review_20261009.md): 기존 valid20 전체 원본영상 실행 79개에서 입력 이미지 SHA와 저장 metrics를 PNG/GT로 대조했다. 동일 마스크를 합치면 독립 후보 69개다. 공식 지표는 각 이미지의 4개 클래스 IoU 평균(빈 union=0)을 20장 평균한 값이다. 릴리스 `Model349_exh14` 80.2312313%를 독립 재현했다. 정답으로 이미지마다 69개 중 최적 마스크를 고르는 **비실용적 상한은 80.9820608%**다. 이는 모델이 아니며 81% 도달 증거가 아니다. 이미지·클래스별 IoU의 상한 81.2306403% 역시 일관된 한 장의 분할 마스크를 뜻하지 않는다. 완료된 후보 중 전체 최고는 결합 TTA 80.2548841%였다. 릴리스 기본 마스크에서 Model401 계열 `matched_source/control`의 예측 라벨 **Al3Ni(0)·Al(2)** 픽셀을 우선하는 고정 규칙을 선택했다. 별도 PNG 20장을 GT 없이 생성한 뒤 PNG/GT 80IoU를 독립 채점해 **80.4746206%**, 릴리스 대비 **+0.2433893%p**를 확인했다. 시편 leave-one-out으로 후보와 규칙을 매번 다시 고른 값은 **80.4087871%**, 릴리스 대비 **+0.1775558%p**다. 최악 A71_step2_sample1은 −0.2300309%p다.
- [decision.json](experiments/model349_compression/accuracy81_candidate_oracle_20261009/decision.json): decision=exploratory_candidate_no_base_change; reason=20-image PNG/GT score improves, and specimen leave-one-out rule selection is positive, but 69 runs × 30 rules were screened on repeatedly used valid labels; there is no independent; baseline_percent=80.23123131980381; candidate_percent=80.47462061710547; delta_pp=0.24338929730166603; worst_image=A71_step2_sample1; worst_image_delta_pp=-0.2300308849183752; specimen_leave_one_out_percent=80.40878714042867; specimen_leave_one_out_delta_pp=0.17755582062488617; all_completed_candidate_image_oracle_percent=80.9820608393505; base_changed=False; test_used=False; independent_generalization_verified=False; goal_81_percent_achieved=False; representative_candidate=stage1_R_plus_E; quality_pass=False
- [review_correction_publication.json](experiments/model349_compression/accuracy81_candidate_oracle_20261009/review_correction_publication.json): commit=d4ef4b0; representative=stage1_R_plus_E; cv_recompute_choices_exact=20; cv_scope=69 runs x30 rules reselected inside specimen exclusion; stage2_F_adopted=False; quality_pass=False; fresh_valid20_test10=running; remote_zip_sha256=cedf0d423a7f15b52ce7b2b364a066499eaa96cd8ba3af16d3eaec529c8e7fd1

### accuracy81_fullimage_iou_20261009

- 작업 폴더: [accuracy81_fullimage_iou_20261009](experiments/model349_compression/accuracy81_fullimage_iou_20261009/)
- [전체 이미지 Soft IoU 학습 결과](experiments/model349_compression/accuracy81_fullimage_iou_20261009/report.md): 부모 출력 재사용 평가이며 전체 fresh 실행이 아니다. train70만 학습하고 emptyunion0 이미지별4클래스평균/20평균을PNG/GT독립채점했다. source snapshot/SHA/학습receipt/native shape/합성hardIoU/zero-refiner20maskexact/원본불변이통과했다. 둘다기각. 각8epochs/560updates,최종epoch8만평가,각checkpoint49,726bytes. native이미지SoftIoU로학습단위를맞추는것만으로개선되지않았다. 악화원인을확정하지않는다. 기존base/릴리스유지/test mIoU없음/81미달/독립일반화미확인.

### accuracy81_fullimage_oof397_20261009

- 작업 폴더: [accuracy81_fullimage_oof397_20261009](experiments/model349_compression/accuracy81_fullimage_oof397_20261009/)
- 로컬 요약 보고서·결정 파일 없음. 실행 완료 여부는 아래 근거와 서버 프로세스로 확인한다.
- [preflight_source_lock.json](experiments/model349_compression/accuracy81_fullimage_oof397_20261009/preflight_source_lock.json)
- [protocol.md](experiments/model349_compression/accuracy81_fullimage_oof397_20261009/protocol.md)

### accuracy81_fullimage_parent14_20261009

- 작업 폴더: [accuracy81_fullimage_parent14_20261009](experiments/model349_compression/accuracy81_fullimage_parent14_20261009/)
- 로컬 요약 보고서·결정 파일 없음. 실행 완료 여부는 아래 근거와 서버 프로세스로 확인한다.
- [preflight_source_lock.json](experiments/model349_compression/accuracy81_fullimage_parent14_20261009/preflight_source_lock.json)
- [protocol.md](experiments/model349_compression/accuracy81_fullimage_parent14_20261009/protocol.md)

### accuracy81_hr336_moments_20261009

- 작업 폴더: [accuracy81_hr336_moments_20261009](experiments/model349_compression/accuracy81_hr336_moments_20261009/)
- [HR336 이미지별 정규화 통계 검증](experiments/model349_compression/accuracy81_hr336_moments_20261009/report.md): 이미지별 4클래스 IoU 평균 후 valid20 평균. 빈 union은 0. 저장 PNG와 GT를 독립 재채점하고 실행 기록의 80개 IoU와 일치함을 확인했다. 가중치 14개, 1,755,428,956 bytes. 원본 가중치·소스와 재사용 확률 파일의 SHA가 유지됐다. HR336 변경 경로만 원본 이미지에서 추론했고 다른 분기는 기존 확률을 재사용했다. GPU 실행 시간은 전체 fresh 앙상블 시간이 아니며 단회 결과로 속도 개선을 판단하지 않는다. 현재 이미지의 특성 통계만 사용하고 추론 중 정답을 읽지 않는다. 같은 valid를 반복 사용한 개발평가이므로 일반화 개선을 보장하지 않는다. test 정답 또는 test mIoU는 없다.
- [decision.json](experiments/model349_compression/accuracy81_hr336_moments_20261009/decision.json): decision=reject_both_keep_existing_base; reason=Both fixed label-free BN moment interpolation candidates reduce official valid image mIoU substantially.; base_percent=80.23122801008668; independent_png_scoring_agrees=True; independent_generalization_verified=False; whole_fresh_pipeline=False; base_release_changed=False; test_used=False; target81_reached=False

### accuracy81_hrboundary_20261008

- 작업 폴더: [accuracy81_hrboundary_20261008](experiments/model349_compression/accuracy81_hrboundary_20261008/)
- [경계 보조 손실 복원 결과](experiments/model349_compression/accuracy81_hrboundary_20261008/boundary_final_report.md): train70만동일부모/seed81336/기존augmentation·LR·BN통계고정/4epoch 조건으로추가학습하고최종epoch4만원본valid20전체재추론했다. 다른13가중치/융합/라우팅/TENT5는유지했다. 20PNG/GT독립재채점은서버metrics의80 IoU와정확히일치했다. 원래base 대비-0.0649795%p, 같은CE/Dice control 대비-0.0193911%p. 최악A74_step3_sample4 -1.0055761%p. boundary손실제외를하락원인으로단정할수없고복원으로도개선이없었다. 420optimizerupdates/학습924.58초,전체valid 단회451.50초. 동일train140파일/부모SHA/seed/segmentationloss·dataset·paletteSHA를검증했다. 원본source/weights불변receipt와종료PID를확인했다. 채택하지않고원래base유지. 반복사용valid의추가개발평가/사후가설이다. 독립일반화/81%달성/test점수를주장하지않는다.
- [HR336 경계 보조 손실 학습 원본 재추론 검증](experiments/model349_compression/accuracy81_hrboundary_20261008/boundary_report.md): 빈 union=0인 이미지별4클래스평균 후20장평균. PNG/GT 독립재채점과실행metrics일치 확인. 동일valid반복개발평가이며 독립일반화 증거가 아니다. 가중치는 원본13개와 추가 학습HR336 1개다. 새 checkpoint bytes는 bundle manifest 개별 row를 합산해야 하며 원본 크기로 단정하지 않는다. 시간은 속도개선 근거가 아니다. 중단 후 재개한 경우 전체 소요시간을 측정하지 못했고 재개 구간만 표시한다. 기존릴리스/원본가중치보존. test mIoU 없음.
- [boundary_decision.json](experiments/model349_compression/accuracy81_hrboundary_20261008/boundary_decision.json): adopted=False; base_unchanged=True; delta_vs_base_pp=-0.06497949645281054; delta_vs_matched_control_pp=-0.01939112376134755; reason=below both base and matched control; independent_generalization_verified=False

### accuracy81_hrcontext_20261009

- 작업 폴더: [accuracy81_hrcontext_20261009](experiments/model349_compression/accuracy81_hrcontext_20261009/)
- [HR336 학습 문맥 확대 검증](experiments/model349_compression/accuracy81_hrcontext_20261009/report.md): 공식 per-image4IoU 평균 후20장평균, emptyunion0. PNG/GT 독립 재채점과 저장 metrics의 모든 IoU 일치를 확인했다. 원본 base 80.2312313%, 기존 512 학습 control 80.1856429%, 이번 1024 학습 80.1820844%. valid 반복 개발평가이며 독립 일반화를 보장하지 않는다. 기존 대조군과 동시 실행은 아니고 학습 픽셀 수·연산량은 4배다. 초기화/손실/학습률/epoch/샘플 수/배치 구조/추론 레시피는 유지했다. 원본과 기존 릴리스를 보존했고 test 정답·test mIoU는 없다. 최종 checkpoint SHA: `74f6843c12be6927ac5420c87a9094fd3eae6fd3b4ca8f950e591e3d1bde1c58`. 4epochs/420updates, 학습 2726.73초. 전체 fresh valid 추론 476.65초(단회, 속도 개선 근거 아님). 가중치14개, 총 1,755,406,329 bytes. 구조·파라미터 수를 줄이지 않았으며 원본 대비 미세한 파일 크기 차이는 저장 형식 차이로 취급한다. 최종 BN1029 buffers와 동결 boundary2 tensors가 원본과 exact이고 모든 state가 finite다.
- [decision.json](experiments/model349_compression/accuracy81_hrcontext_20261009/decision.json): decision=reject_context_keep_current_base; reason=Raw fresh valid20 result is below both original base and previous matched-training control; no demonstrated improvement from larger training context.; miou_percent=80.18208443240248; vs_base_pp=-0.04914688740133011; vs_control_pp=-0.003558514709867122; whole_fresh_pipeline=True; independent_png_scoring_agrees=True; original_preserved=True; test_used=False; target81_reached=False; independent_generalization_verified=False
- [training_completion.json](experiments/model349_compression/accuracy81_hrcontext_20261009/training_completion.json): arm=context; epochs=4; optimizer_updates=420; elapsed_seconds=2726.7252570469864; peak_cuda_allocated_bytes=11391257600; final_checkpoint=/workspace/model349_compress/accuracy81_hrcontext_20261009/training/context/epoch4.pt; final_sha256=74f6843c12be6927ac5420c87a9094fd3eae6fd3b4ca8f950e591e3d1bde1c58; parent_unchanged=True; valid_read=False; test_read=False

### accuracy81_hrlovasz_20261008

- 작업 폴더: [accuracy81_hrlovasz_20261008](experiments/model349_compression/accuracy81_hrlovasz_20261008/)
- [cancelled_for_idle_gpu3.json](experiments/model349_compression/accuracy81_hrlovasz_20261008/cancelled_for_idle_gpu3.json): reason=GPU3 now idle; preserve exact paired recipe in new root and execute concurrently with TTA, avoid duplicate queued training; time=1791462713.5550458

### accuracy81_hrlovasz_gpu3_20261008

- 작업 폴더: [accuracy81_hrlovasz_gpu3_20261008](experiments/model349_compression/accuracy81_hrlovasz_gpu3_20261008/)
- [HR336 추가 학습 대조군 결과](experiments/model349_compression/accuracy81_hrlovasz_gpu3_20261008/control_report.md): 원본 HR336 부모에서 train70만 4 epoch 추가 학습한 최종 checkpoint다. 원본 가중치13개와 계수/라우팅/TENT5는 그대로다. 20개 원본 valid PNG의 독립 재채점은 실행 metrics의 80개 IoU와 일치했다. 빈 union=0인 이미지별 4클래스 IoU 평균 후 20장 평균을 적용했다. 평균 -0.0455884%p, 최악 A74_step3_sample4 -1.1201246%p. 개선 11장 / 악화 9장. 조건부 시편 그룹 bootstrap 95% 구간 [-0.2112787, +0.0900266]%p. 반복 사용한 valid의 사후 진단이며 독립 일반화 근거가 아니다. 학습 420 optimizer updates / 899.49초. 전체 valid 단회 추론 452.21초(GPU3). 새 14개 checkpoint 총 1,755,406,329 bytes이며 원본 부모 SHA 불변을 확인했다. 속도 비교 근거가 아니다. 추가 학습 자체는 개선을 보여주지 못했고 base 교체를 기각한다. Lovász 처리군은 동일 부모/seed/데이터/augmentation/4 epoch 정책으로 진행 중이다. 처리군이 대조군보다 높더라도 원래 base보다 높아야 성능 향상 후보가 된다. 원본 base 및 기존 릴리스는 유지하며 test mIoU는 없다.
- [HR336 대조 학습 원본 재추론 검증](experiments/model349_compression/accuracy81_hrlovasz_gpu3_20261008/learning_report.md): 빈 union=0인 이미지별4클래스평균 후20장평균. PNG/GT 독립재채점과실행metrics일치 확인. 동일valid반복개발평가이며 독립일반화 증거가 아니다. 가중치는 원본13개와 추가 학습HR336 1개다. 새 checkpoint bytes는 bundle manifest 개별 row를 합산해야 하며 원본 크기로 단정하지 않는다. 시간은 속도개선 근거가 아니다. 중단 후 재개한 경우 전체 소요시간을 측정하지 못했고 재개 구간만 표시한다. 기존릴리스/원본가중치보존. test mIoU 없음.
- [HR336 Lovász 대조 실험 최종 결과](experiments/model349_compression/accuracy81_hrlovasz_gpu3_20261008/lovasz_report.md): 동일 원본 부모·train70·seed·crop/augmentation·4epoch 조건으로 CE/Dice와 Lovász를 비교했다. 양쪽 최종 epoch4(420 optimizer updates)만 원본 valid20 전체 재추론했다. 다른13개가중치/융합/라우팅/TENT5는유지했다. 20개 PNG/GT 독립재채점에서80 IoU가실행metrics와정확히일치한다. 빈union=0의이미지별4클래스평균후20장평균이다. Lovász의 matched control 대비 평균 -0.0049840%p, 조건부19그룹 bootstrap95% [-0.0294774, +0.0208030]%p. 원래base 대비 최악 A74_step3_sample4 -0.9763981%p. 반복valid평가의사후조건부구간이며독립일반화증거가아니다. Lovász 학습 956.01초, 최종14 checkpoint총 1,755,406,329bytes, 전체valid 단회 추론 475.42초(GPU3). 부모SHA불변/학습중valid·test미열람/동일train140파일SHA·부모SHA·seed를검증했다. 원본pipeline의training-onlyboundary손실을양쪽모두제외했으므로원래HR336전체학습레시피의연장이아니다.
- [control_review.json](experiments/model349_compression/accuracy81_hrlovasz_gpu3_20261008/control_review.json): status=rejected_for_base_replacement; miou_percent=80.18564294711234; delta_pp=-0.045588372691462986; worst_image=A74_step3_sample4; worst_image_delta_pp=-1.1201245892626055; independent_generalization_verified=False; training_epochs=4; optimizer_updates=420; original_parent_preserved=True; test_score_available=False; checkpoint_bytes=1755406329; lovasz_pending=True
- [lovasz_review.json](experiments/model349_compression/accuracy81_hrlovasz_gpu3_20261008/lovasz_review.json): adopted=False; base_unchanged=True; lovasz_miou_percent=80.18065893496693; vs_parent_base_delta_pp=-0.05057238483687776; vs_matched_control_delta_pp=-0.0049840121454147734; training_files_parent_seed_exact=True; independent_generalization_verified=False; reason=both trained arms below original base; Lovasz below matched control; test_score_available=False

### accuracy81_hrscale_20261008

- 작업 폴더: [accuracy81_hrscale_20261008](experiments/model349_compression/accuracy81_hrscale_20261008/)
- [HR336 배율 component replay 검증](experiments/model349_compression/accuracy81_hrscale_20261008/scale_report.md): 빈 union=0인 이미지별4클래스평균 후20장평균. PNG/GT 독립재채점과실행metrics일치 확인. 동일valid반복개발평가이며 독립일반화 증거가 아니다. 가중치14개/1,755,428,956바이트 유지. HR336만 새 입력 배율로 원본 재추론하고 변경 없는6분기의SHA검증된확률을재사용했다. 전체fresh재추론/독립일반화성적이 아니다. 시간은 신규component 또는CPU평균구간만이며 전체모델시간/속도개선근거가 아니다. 기존릴리스/원본가중치보존. test mIoU 없음. 같은 재사용 분기/실행 환경의 대조군은80.2312297%다. 표의 canonical 원본80.2312313%는 별도2GPU 실행으로 아주 작은 수치 차이가 있다. 아래는 동일 replay 대조군 대비 비교다. 모든후보가낮아채택하지않는다. 입력배율과native문맥이학습배율2와달라생기는오류가커졌다는쪽의결과이며 학습을배율에맞춰다시수행한처치까지기각한것은아니다.
- [scale_decision.json](experiments/model349_compression/accuracy81_hrscale_20261008/scale_decision.json): adopted=False; base_unchanged=True; reason=all three fixed scale candidates below replayed original scale2 baseline; baseline_replay_percent=80.23122969597495; component_replay_not_whole_fresh_inference=True; independent_generalization_verified=False; test_score_available=False

### accuracy81_model392_pixel_oof_20261009

- 작업 폴더: [accuracy81_model392_pixel_oof_20261009](experiments/model349_compression/accuracy81_model392_pixel_oof_20261009/)
- [decision.json](experiments/model349_compression/accuracy81_model392_pixel_oof_20261009/decision.json): decision=screen_fail; reason=No fixed blend reached +0.50 pp on heldout fold4; no valid/GPU follow-up; result_sha256=e60868c91c30f1f65edece35492aee07a8c4f67fd65db65aad5e6e200297e084; best_delta_pp=0.10835649573246453; valid_read=False; test_read=False; strict_generalization=False

### accuracy81_oof_gray_ablation_20261009

- 작업 폴더: [accuracy81_oof_gray_ablation_20261009](experiments/model349_compression/accuracy81_oof_gray_ablation_20261009/)
- [OOF 밝기 입력 고정 비교 결과](experiments/model349_compression/accuracy81_oof_gray_ablation_20261009/report.md): 확률을 재사용한 개발 평가다. 전체 fresh 실행 시간이 아니며 독립 일반화 증거가 아니다. train70만학습/valid20PNG와GT독립공식채점/emptyunion0 확인. 둘다기각. 마지막epoch8만평가했고각8epochs/1120updates,checkpoint/sourceSHA,zero-refiner20maskexact,원본확률과14parent불변을확인했다. 밝기정규화/제거가해결책이라는증거없음. 기존base/릴리스유지/test mIoU없음/81미달.

### accuracy81_oof_nogray_20261009

- 작업 폴더: [accuracy81_oof_nogray_20261009](experiments/model349_compression/accuracy81_oof_nogray_20261009/)
- 로컬 요약 보고서·결정 파일 없음. 실행 완료 여부는 아래 근거와 서버 프로세스로 확인한다.
- [preflight_source_lock.json](experiments/model349_compression/accuracy81_oof_nogray_20261009/preflight_source_lock.json)
- [protocol.md](experiments/model349_compression/accuracy81_oof_nogray_20261009/protocol.md)

### accuracy81_oof_refiner_20261009

- 작업 폴더: [accuracy81_oof_refiner_20261009](experiments/model349_compression/accuracy81_oof_refiner_20261009/)
- [같은 입력에서의 OOF 잔차 평가](experiments/model349_compression/accuracy81_oof_refiner_20261009/matched_report.md): 원본 확률을 재사용한 탐색이다. 전체fresh 추론/독립일반화 성적이 아니다. 공식 emptyunion0 이미지별4클래스평균/20평균을PNG/GT독립채점했고metrics와일치한다. 같은 source 대비 -0.1733055%p. 후보기각. 최악A74_step3_sample4 -1.3974390%p. zero-refiner20maskexact와원본확률/parent14sourceweights불변을확인했다. Model397 OOF 네트워크 계열을맞춰도개선되지않았으므로앞선base14전이손실을입력차이만으로설명할수없다. 원인을확정하지않는다. 새학습/test사용없음. 기존base/릴리스유지/81미달.
- [작은 잔차 보정 모델 원본 valid20 검증](experiments/model349_compression/accuracy81_oof_refiner_20261009/residual_report.md): 빈 union=0인 이미지별4클래스평균 후20장평균. PNG/GT 독립재채점과실행metrics일치 확인. 동일valid반복개발평가이며 독립일반화 증거가 아니다. 원래14가중치는보존하고11,876parameter 잔차모델1개를추가한다. checkpoint bytes는training receipt를검증해별도기록한다. refined elapsed_seconds는부모출력재사용후보정평가만이며전체fresh시간이아니다. 시간은단회이고속도개선근거가아니다. train70 teacher 자료는부모가학습한이미지여서held-out/OOF가아니다. 같은valid반복개발평가/독립일반화미확인. 기존릴리스/원본보존/test mIoU 없음. 후보를 기각한다. PNG/GT 독립 공식채점79.9993891386%, 같은 부모80.2312280101% 대비−0.2318388715%p. 부모출력은재사용했으므로전체fresh새실행이아니다. 모델397의rawOOF학습입력과base14의배포logits차이가있으며,그차이가악화원인인지이결과만으로단정하지않는다. 원본14source/weightsSHA불변,zero-refiner20maskexact,8epochs/1120updates와train70정답만사용한학습을확인했다.
- [decision.json](experiments/model349_compression/accuracy81_oof_refiner_20261009/decision.json): adopt=False; goal81_achieved=False; raw_parent_reused=True; candidate_whole_fresh_pipeline=False; official_png_miou_percent=79.99938913855537; same_parent_miou_percent=80.23122801008668; delta_pp=-0.2318388715313202; worst_image=A71_step2_sample1; worst_image_delta_pp=-2.262182014967995; positive_images=5; negative_images=15; model_and_data_unchanged_vs_unweighted_control_except_oof_inputs=True; parent_source_weights_unchanged=True; training_source_vs_deployment_parent_differ=True; independent_generalization_verified=False

### accuracy81_oof_zgray_20261009

- 작업 폴더: [accuracy81_oof_zgray_20261009](experiments/model349_compression/accuracy81_oof_zgray_20261009/)
- 로컬 요약 보고서·결정 파일 없음. 실행 완료 여부는 아래 근거와 서버 프로세스로 확인한다.
- [preflight_source_lock.json](experiments/model349_compression/accuracy81_oof_zgray_20261009/preflight_source_lock.json)
- [protocol.md](experiments/model349_compression/accuracy81_oof_zgray_20261009/protocol.md)

### accuracy81_p100_exh14_transfer_20261009

- 작업 폴더: [accuracy81_p100_exh14_transfer_20261009](experiments/model349_compression/accuracy81_p100_exh14_transfer_20261009/)
- [같은 P100 가중치의 원래 route와 80.2% 방법 이식 비교](experiments/model349_compression/accuracy81_p100_exh14_transfer_20261009/matched_report.md): 원본영상 fresh 추론 20장 PNG/GT의 80개 IoU를 두 P100 경로에서 독립 채점했다. 공식 image별4class 평균, empty union=0. 같은 P100 가중치에서 route 이식 변화 -0.0886312%p. 릴리스 가중치 exh14 대비 P100 이식 변화 -0.5120749%p. 같은 실행기의 원래 route가 과거 P100 캐시 점수와 다른 정도 -0.1017313%p. 소스 감사를 통해 조건 차이를 확인했다: 과거 P100 B4 valid 확률은 D4, HRNet valid 확률은 flip4 TTA로 생성됐으나 이 새 대조군 실행기는 양쪽 모두 `tta=False`였다. 설정 파일의 TTA 이름과 달리 실제 실행기의 인자가 우선한다. 각 차이의 기여량과 다른 가능한 원인은 분리 측정하지 않았다. 코드·서버 SHA 증거는 `fresh_cache_gap_audit.json`에 있다. 판정: 이식 구성을 기각하고 기존 릴리스 base를 유지한다. 두 fresh 실행의 입력·실행기·가중치 출처와 독립 채점은 확인됐다. -0.0886312%p는 **이번에 실행한 두 구성의 관측 차이**이며, 과거 Model349 파이프라인의 TTA 포함 route만 바꿨을 때의 순수 인과 효과로 단정하지 않는다. 릴리스와 P100의 차이도 체크포인트가 다르므로 GPU 단독 효과로 해석하지 않는다.
- [80.2% 방법의 P100 가중치 이식 결과](experiments/model349_compression/accuracy81_p100_exh14_transfer_20261009/report.md): 원본영상 valid20을 fresh 추론하고 PNG/GT 20장 80개 IoU를 독립 채점했다. 공식 image별4class 평균, empty union=0. 릴리스 가중치 exh14 80.2312313%, P100 가중치 이식 79.7191564%, 변화 -0.5120749%p. 과거 P100 원형 Model349 79.9095189%와의 수치 차이 -0.1903625%p는 참고값이다. 이 비교에는 route와 과거 캐시 구현 차이가 함께 있다. 14개 P100 checkpoint SHA/bytes 및 이식 추론 코드·route 동일성 확인. 전체 fresh 단회 879.78초. 반복 사용한 valid20 결과이므로 독립 일반화 또는 GPU만의 효과가 아니다. test 정답은 사용하지 않았다.
- [checkpoint_geometry_decision.json](experiments/model349_compression/accuracy81_p100_exh14_transfer_20261009/checkpoint_geometry_decision.json): checkpoint_count=14; cpu_only=True; full_checkpoint_linear_soup_screen=no_go_without_functional_compatibility_test; reason=All 14 corresponding checkpoints differ. B4 relative L2 is 0.448-0.660 and HRNet relative L2 is 0.775-0.816; naive linear interpolation can cross incompatible weight basins. This g; followup_gpu_started=False; base_changed=False; test_used=False
- [decision.json](experiments/model349_compression/accuracy81_p100_exh14_transfer_20261009/decision.json): decision=reject_p100_exh14_transfer; reason=Independent fresh valid20 PNG scoring fell below both the matched fresh P100 original-route control and the release exh14 baseline.; metric=image-wise four-class mean IoU over 20 validation images; empty union=0; p100_transfer_percent=79.71915639973314; p100_original_matched_percent=79.80778760741404; transfer_vs_matched_pp=-0.08863120768091148; release_exh14_percent=80.23123131980381; transfer_vs_release_pp=-0.5120749200706709; historical_p100_cached_percent=79.90951887388398; fresh_control_vs_historical_cached_pp=-0.10173126646994035; historical_reproduction_gap_unresolved=False; interpretation=The two fresh configurations use the same executable, P100 checkpoint source, no-TTA implementation, and inputs; the transfer measures 0.0886312 percentage points lower. Historical; independent_png_gt_scoring=True; test_used=False; independent_generalization_verified=False; base_changed=False; release

### accuracy81_p100_mask_consensus_20261009

- 작업 폴더: [accuracy81_p100_mask_consensus_20261009](experiments/model349_compression/accuracy81_p100_mask_consensus_20261009/)
- [decision.json](experiments/model349_compression/accuracy81_p100_mask_consensus_20261009/decision.json): decision=reject_screen; reason=Specimen leave-one-out score is below the release baseline; full-valid gain is only 0.0185 percentage points and does not pass the fixed +0.5 point screen gate.; release_percent=80.23123131980378; best_full_percent=80.24974891690124; best_full_delta_pp=0.01851759709745604; specimen_leave_one_out_percent=80.21076449063702; specimen_leave_one_out_delta_pp=-0.020466829166765343; gpu_followup=False; base_changed=False; test_used=False; independent_generalization_verified=False

### accuracy81_residual_20261008

- 작업 폴더: [accuracy81_residual_20261008](experiments/model349_compression/accuracy81_residual_20261008/)
- [작은 잔차 보정 모델 원본 valid20 검증](experiments/model349_compression/accuracy81_residual_20261008/residual_report.md): 빈 union=0인 이미지별4클래스평균 후20장평균. PNG/GT 독립재채점과실행metrics일치 확인. 동일valid반복개발평가이며 독립일반화 증거가 아니다. 원래14가중치는보존하고11,876parameter 잔차모델1개를추가한다. checkpoint bytes는training receipt를검증해별도기록한다. refined elapsed_seconds는전체fresh부모valid+feature추출+잔차평가이며학습시간을제외한다. 시간은단회이고속도개선근거가아니다. train70 teacher 자료는부모가학습한이미지여서held-out/OOF가아니다. 같은valid반복개발평가/독립일반화미확인. 기존릴리스/원본보존/test mIoU 없음. 후보는 기각한다. 같은 fresh parent 대비 −0.4950325%p이며,81% 목표를 달성하지 못했다. 초기 zero-refiner 20장 exact 검사를 통과해 입력 추출 오류와 모델의 실제 변화는 구분했다. 원본 source/14가중치는 변경되지 않았고 base pointer도 그대로다. 9장 개선/11장 악화. 총15 checkpoint 1,755,478,682bytes; 보정 모델49,726bytes. 전체fresh valid pipeline521.2025초,같은parent450.7893초(모두단회). 속도개선근거가 아니다. test10은 이 기각 후보에 실행하지 않았으며 test 성능을 주장하지 않는다.
- [residual_decision.json](experiments/model349_compression/accuracy81_residual_20261008/residual_decision.json): adopt=False; reason=Independently verified valid20 degradation vs same fresh parent; target81 not reached; parent_miou_percent=80.23122801008668; candidate_miou_percent=79.7361955387584; delta_pp=-0.49503247132829387; worst_image=A71_step2_sample1; worst_image_delta_pp=-5.3888960357520554; positive_images=9; negative_images=11; total_checkpoint_bytes=1755478682; base_pointer_changed=False; parent_source_weights_unchanged=True; independent_generalization_verified=False; goal81_achieved=False

### accuracy81_residual_balanced_20261008

- 작업 폴더: [accuracy81_residual_balanced_20261008](experiments/model349_compression/accuracy81_residual_balanced_20261008/)
- [작은 잔차 보정 모델 원본 valid20 검증](experiments/model349_compression/accuracy81_residual_balanced_20261008/residual_report.md): 빈 union=0인 이미지별4클래스평균 후20장평균. PNG/GT 독립재채점과실행metrics일치 확인. 동일valid반복개발평가이며 독립일반화 증거가 아니다. 원래14가중치는보존하고11,876parameter 잔차모델1개를추가한다. checkpoint bytes는training receipt를검증해별도기록한다. refined elapsed_seconds는부모출력재사용후잔차평가만이며전체fresh시간이아니다. 시간은단회이고속도개선근거가아니다. train70 teacher 자료는부모가학습한이미지여서held-out/OOF가아니다. 같은valid반복개발평가/독립일반화미확인. 기존릴리스/원본보존/test mIoU 없음. 클래스 균형 후보는 기각한다. 독립 PNG/GT 채점79.2159837592%,같은원본80.2312280101% 대비−1.0152442508%p. 원본예측을읽기전용재사용한평가이며전체fresh새실행은아니다. 실제모델학습8epochs/1120updates와zero-refiner20마스크exact,원본코드/가중치SHA불변을확인했다. 클래스불균형을단순히가중CE로바꾸는방법은이실험에서개선되지않았다. 오류의원인이클래스불균형인지단정할수없다. train70 빈도와 CE 가중치: [0.25140569863715256, 0.09683654638828977, 0.5791192148191883, 0.07263854015536968] / [0.7796479353319271, 1.2562181596093633, 0.51369175284936
- [decision.json](experiments/model349_compression/accuracy81_residual_balanced_20261008/decision.json): adopt=False; goal81_achieved=False; raw_parent_reused=True; candidate_whole_fresh_pipeline=False; official_png_metric_percent=79.21598375924209; vs_same_parent_delta_pp=-1.0152442508445936; worst_image=A71_step2_sample1; worst_image_delta_pp=-8.233773389078753; parent_source_weights_unchanged=True; base_changed=False; generalization_verified=False

### accuracy81_stage1_fresh_20261009

- 작업 폴더: [accuracy81_stage1_fresh_20261009](experiments/model349_compression/accuracy81_stage1_fresh_20261009/)
- [완료된 R 중간 검증](experiments/model349_compression/accuracy81_stage1_fresh_20261009/completed_R_summary.md): R valid20만 완료돼 별도로 채점했다. 전체 1단계·test 실험은 미완료다. - 새 R: 80.2309863330% - 기존 canonical R: 80.2312313198% - 차이: −0.0002449868%p / 17픽셀 - 입력 SHA 일치, archive SHA·안전 추출 및 PNG/GT 80개 IoU 독립 검증 완료 수치 차이의 원인은 단정하지 않는다. 전체 후보는 같은 실행 R와 기존 canonical 모두에 비교한다.
- [1단계 전체 valid 재추론 및 독립 채점](experiments/model349_compression/accuracy81_stage1_fresh_20261009/completed_valid_summary.md): 원본 valid20 영상에서 고정 R·E 가중치를 다시 추론하고 1단계 규칙으로 생성한 PNG 20장을 로컬 GT로 독립 채점했다. 아카이브 SHA256 검증 및 입력·보정 SHA 일치를 확인했다. - fresh R: 80.2309863330% - fresh 1단계 R+E: 80.4744236018% - fresh R 대비: +0.2434372688%p - 기존 저장 1단계 대비: -0.0001970153%p, 7 / 15,757,260 픽셀 차이 - fresh R 대비 최악 이미지: A71_step2_sample1, -0.2304701319%p, 기존 품질 기준 통과 여부: False 이는 **고정 가중치의 추론 재현 검증**이다. 처음부터 재학습한 성능의 재현성을 입증하지 않는다. valid20 반복 사용 및 기존 선택 편향은 남는다. 시험10 결과는 아직 미완료이며 전체 파이프라인 완료로 기록하지 않는다. 81% 목표는 미달성이다.
- [1단계 R + E 원본 영상 재추론 결과](experiments/model349_compression/accuracy81_stage1_fresh_20261009/report.md): 원본 영상부터 R의 릴리스 14가중치 및 E의 B4 4-fold/HRNet 2-seed × 4-fold를 다시 계산했다. F와 확신도 제한은 사용하지 않았다. 원본 소스·가중치 SHA는 불변이며, archive SHA/안전 추출 및 PNG/GT 20장 80개 IoU 독립 채점을 통과했다. - 새 1단계 valid20: **80.4744236018%** - 같은 실행의 R: 80.2309863330% - 같은 실행 R 대비: +0.2434372688%p - 기존 canonical R 대비: +0.2431922820%p - 저장 부모 출력 기반 후보 대비: -0.0001970153%p - 최악 영상: A71_step2_sample1, -0.2300308849%p - 기존 영상별 손실 기준 통과: False test10은 새 R 대비 161,731 / 9,074,964 픽셀이 달라, 일치율 98.21783315%다. 기존 canonical exh14 대비 일치율은 98.21780009%다. 영상별 변화 및 클래스 면적 비율은 comparison.json에 기록했다. test에는 정답이 없어 mIoU 또는 성능 증감을 주장하지 않는다. 1단계 시편 제외 80.4088%는 69개 실행 × 30개 규칙 전체를 매번 다시 선택한 값이다. 후보와 R 자체의 과거 valid 기반 개발 선택 편향까지 제거하지는 못한다. F는 조건부 CV에서 1단계 대비 손해로 미채택이다. 기존 base는 유지하며, 81% 목표 달성 여부는 별도 전체 요구

### audit_20261008

- 작업 폴더: [audit_20261008](experiments/model349_compression/audit_20261008/)
- [Model349 경량화 재개 감사 (2026-10-08)](experiments/model349_compression/audit_20261008/README.md): 이 디렉터리는 기존 릴리스, 가중치, 미완료 산출물을 덮어쓰지 않는 추가 감사 실험이다. 한국어 최종 결과는 `report.md`, 기계 판독 결과는 `comparison.json`, `batched_summary.json`, `hramp_summary.json`, `batched_test_parity.json`, `hramp_test_parity.json`에 기록한다. 작업 디렉터리의 GPU 사용자 매뉴얼 PDF 전체를 pdftotext로 읽었다. 공유 GPU/스토리지, /workspace 사용 지침을 따른다. GPU6은 다른 컨테이너 작업이 점유하여 사용하지 않고, 비어 있던 GPU7 P100 한 장만 사용한다. 프로토콜 변경은 결과를 보기 전에 `protocol.md`, `cpu_threads_protocol.md`에 기록했다. 원래 FP16 5-step의 모든 component 20장 추론은 완료되었고 fusion에서 마스크 1장만 저장돼 있었다. 기존 확률과 로그는 보존하고 CPU fusion을 별도 경로로 복구했다. 복구 공식 점수는 80.0655029%이며 full walltime은 알 수 없다. TENT3/12의 과거 완료·부분 산출물은 없었다. 기존 PID 기반 대기 스크립트 대신 flock, 현행 작업 확인, GPU 메모리·점유 재확인을 사용한다.
- [Model349 경량화 재개 감사 결과 (2026-10-08)](experiments/model349_compression/audit_20261008/report.md): D4 배치 후보는 사전 탐색 채택 기준을 통과하지 못했다. 저장 크기 개선과 성능·시간 판정은 구분한다. valid20 원본 영상 재추론과 반복 시간의 재현 결과다. 이 valid를 반복 사용했으므로 독립 일반화 검증이나 test 점수 유지 근거로 해석하지 않는다. 기존 릴리스·원본 가중치·미완료 산출물은 덮어쓰지 않았다. - GPU 사용자 매뉴얼 전체를 읽었다. GPU6은 타 컨테이너의 작업(PID 549152, 3328 MiB)이 점유하여 사용하지 않았다. 비어 있던 P100 GPU7 한 장에서 기준과 후보를 순차 실행했다. - GPU6 지정 기존 프로토콜을 GPU7의 동시기 기준 비교로 보완했으며, 결과 전 문서에 기록했다. flock·현재 프로세스 확인·GPU 점유 재확인으로 중복을 막았다. 다른 사용자 작업은 조작하지 않았다. - 과거 FP16 TENT5는 5개 component 모두 20장·100개 확률을 완료했고 fusion에서 마스크 1장만 남아 있었다. 별도 경로로 CPU fusion을 복구했다(80.0655029%). 전체 시간은 알 수 없으므로 null이며 속도 근거에서 제외한다. 중단 원인은 로그만으로 확정하지 못했다. - 과거 TENT3/12에는 완료·부분 산출물이 없었다. 새 경로에 원본 영상부터 실행했다. 이전 고정 PID 대기 큐는 재사용하지 않았다. - 초기 lite15는 614.73초였지만 CPU load와 width4000 B4 지연이 컸다. 이 실행과 과거 366.56초 단회만으로 후보
- [recovered_completion.json](experiments/model349_compression/audit_20261008/recovered_completion.json): images=20; elapsed_seconds=None; fusion_recovery_seconds=30.483100302051753; recovered_from=/workspace/model349_compress/runs/lite14_fp16_valid20; timing_valid=False; mean_image_miou=0.8006550291285413

### base12_best14_mean_20261008

- 작업 폴더: [base12_best14_mean_20261008](experiments/model349_compression/base12_best14_mean_20261008/)
- [base12 + best14 1:1 확률 평균 결과](experiments/model349_compression/base12_best14_mean_20261008/report.md): 공통 후처리 직전 최종 융합 확률을 1:1 평균했다. 기존 확률 캐시를 쓴 CPU 재생이며 원본 영상 추론/독립 일반화 검증이 아니다. 양 끝 단독 구성의 20장 마스크와 IoU가 이전 재채점과 정확히 같음을 확인했다. 비율 탐색은 하지 않았다. base12 대비 평균 +0.0461233%p, 최악 -0.1742671%p (A181_step1_sample1). 품질 기준 통과: False. 평균 손실≤0.01%p/최악 손실≤0.10%p. 상승 12장, 악화 8장. 고유 가중치가 16개로 늘어 경량화 후보가 아니다. 공통 경로를 재사용한 실제 코드/속도는 검증하지 않았다. 80.2 후보 자체가 같은 valid로 선택되었고 평균판 평가도 동일 valid이므로, 품질 기준을 통과하더라도 신규 시편 개선을 증명하지 않는다. 모델 파일 tensor 평균이나 두 mIoU 점수의 산술 평균과 구분한다. 기존 base/릴리스는 유지한다.

### base12_followup_20261008

- 작업 폴더: [base12_followup_20261008](experiments/model349_compression/base12_followup_20261008/)
- [12개 base 후속 경량화·성능 탐색 결과](experiments/model349_compression/base12_followup_20261008/report.md): 세 후보 변경을 분리하고 결과 전 범위를 고정했다. 모든 결과는 반복 사용한 valid20의 탐색·구현 재현이며 독립 일반화 검증이 아니다. 기존 base 및 가중치는 그대로 보존했다. gate: base12 대비 평균 손실 ≤0.01%p 및 최악 이미지 손실 ≤0.10%p. lite15 대비 원래 gate도 JSON에 별도로 기록했다. 클래스별·이미지별 손실, 원본349 비교는 `raw_comparison.json`과 `comparison.json`을 확인한다. 기존 base 확률을 고정한 탐색 채점이다. gamma0.5의 base 점수는 정확히 재현했다. 아래 둘은 국소 손실 gate 초과로 기각했으므로 원본 영상 재추론 후보로 올리지 않았다. Al3Ni 클래스 계수를 올려도 `replace_classes`가 네 클래스 확률을 다시 정규화하므로 다른 클래스가 독립적으로 보존되는 것은 아니다. gamma0.6에서 A153 Eutectic Si IoU는 약 −0.770925%p로 악화했다. 평균만 높아진 조합을 성능 개선판으로 채택하지 않았다. 실행 전 동결한 선택 코드에서는 base12와 기존 lite15의 gate를 모두 통과해야 반복 시간/후속 test 후보로 올리도록 구현했다. 프로토콜 본문의 “품질 통과”는 이 이중 조건을 명확하게 풀어 쓰지 않아 문구가 불완전했다. TENT4는 base12 기준만 통과했고 lite15 대비 A74에서 −0.1177126%p로 기존 국소 손실 한도를 넘었다. 따라서 이번 단계에서
- [completion.json](experiments/model349_compression/base12_followup_20261008/completion.json): diagnostics_completed=True; base_and_weights_unchanged=True; time_candidate=None; independent_generalization_verified=False; new_training_started=False

### base12_highflip4_20261008

- 작업 폴더: [base12_highflip4_20261008](experiments/model349_compression/base12_highflip4_20261008/)
- [고해상도 4방향 예측 결과](experiments/model349_compression/base12_highflip4_20261008/report.md): TENT5·원본/밝기 보정 두 입력·모든 12가중치·계수·후처리를 유지하고 high D4 8방향만 대칭 flip4 4방향으로 줄였다. 다른 기각 후보와 조합하지 않았다. 12개/683,320,900바이트 저장량은 동일하다. 이미지별 4 IoU 평균을 20장 평균, absent/empty=0. PNG/GT 독립 재채점과 metrics 일치를 확인했다. 클래스별·이미지별 상세는 comparison.json에 보관한다. 평균 손실 ≤0.01%p·최악 손실 ≤0.10%p를 두 기준에 모두 적용한다. 품질 조건을 초과하여 반복 시간 비교와 test10을 시작하지 않았다. 후보 단회 시간은 진단 기록이며 확정 속도 개선으로 주장하지 않는다. GPU7 복수 compute 감지 0 frames. 기존 공유 CPU 작업을 보존했고 부하 이력도 보관했다. 같은 valid20에서 반복 탐색한 결과다. 신규 데이터·독립 일반화 증거가 없다. base12는 교체하지 않고 원본 가중치·기존 릴리스도 보존했다.
- [completion.json](experiments/model349_compression/base12_highflip4_20261008/completion.json): diagnostics_completed=True; base_and_weights_unchanged=True; time_candidate=None; independent_generalization_verified=False; new_training_started=False

### base12_inputcache_20261008

- 작업 폴더: [base12_inputcache_20261008](experiments/model349_compression/base12_inputcache_20261008/)
- [12개 base B4 입력 재사용 결과](experiments/model349_compression/base12_inputcache_20261008/report.md): 가중치·모델 순서·출력 결합은 그대로 두고 동일 영상·동일 전처리 설정의 B4 입력만 fold 사이 재사용했다. 원래 base는 변경하지 않았다. 체크포인트 12개/683,320,900바이트는 같다. 정확도 상승 또는 모델 수 감소 실험이 아니다. - 326: 출력 tensor 정확히 동일=True, 최대 오차=0.0; 전처리 호출 2→1. - 329: 출력 tensor 정확히 동일=True, 최대 오차=0.0; 전처리 호출 4→1. - valid20 전체 저장 확률 326: 20장 모두 기존 base와 정확히 동일=True. - valid20 전체 저장 확률 329: 20장 모두 기존 base와 정확히 동일=True. - valid20 전체 저장 확률 336: 20장 모두 기존 base와 정확히 동일=True. - valid20 전체 저장 확률 269: 20장 모두 기존 base와 정확히 동일=True. - cache12_valid20: valid20 image mIoU 80.1255844%, base 대비 평균 +0.0000233%p / 최악 -0.0010261%p, gate True. - timing_1_cache12: valid20 image mIoU 80.1254589%, base 대비 평균 -0.0001021%p / 최악 -0.0016411%p, gate True. - timing_1_control12: valid20 image mIoU 80.1256273%, base 대비 평균 +0.0000663%p / 최악
- [completion.json](experiments/model349_compression/base12_inputcache_20261008/completion.json): diagnostics_completed=True; base_and_weights_unchanged=True; time_candidate=cache12; independent_generalization_verified=False; new_training_started=False

### base12_partial_distill329_20261008

- 작업 폴더: [base12_partial_distill329_20261008](experiments/model349_compression/base12_partial_distill329_20261008/)
- [cancelled_before_launch.json](experiments/model349_compression/base12_partial_distill329_20261008/cancelled_before_launch.json): reason=user switched to Model349_exh14 before pilot launch; training_started=False

### base12_resident_packed_20261008

- 작업 폴더: [base12_resident_packed_20261008](experiments/model349_compression/base12_resident_packed_20261008/)
- [추가 탐색 실행·재현 안내](experiments/model349_compression/base12_resident_packed_20261008/README.md): 이 폴더는 기존 `Model349_lite12_base`와 분리된 실험이다. 채택 여부는 `report.md`/`comparison.json`의 공식 품질과 반복 시간 조건으로 판단한다. 완료 전 로그나 단회 값으로 채택하지 않는다. 서버 위치: `/workspace/model349_compress/base12_resident_packed_20261008`. - `protocol.md`: 실행 전 후보/선택/판정 조건. - `run_experiment.py`: GPU7 점유/잠금 확인 → 후보 준비 → 상태 복원 probe → 원본 valid 후보 → 조건부 무손실 결합 → 선택 후보와 기준 각각 3회 교차 추론 → test10. - `probe.py`: 실제 GPU에서 source 전환 및 고의 변조 후 전체 원본 parameter/buffer/grad/eval 복원 확인. - `finalize.py`: 완료된 실행의 PNG/GT 독립 채점, 클래스·이미지별 손실, component/전체 시간, 부하/복원/저장 감사, 결과 아카이브. - `bundles/*/source_hashes.json` 및 `checkpoint_manifest.json`: 실행한 소스·설정 및 실제 가중치 SHA/바이트. `soup_provenance.json`은 합성 출처, `lossless_storage_verification.json`은 ZIP 및 tensor 동일성 기록.
- [12개 base 추가 탐색 결과](experiments/model349_compression/base12_resident_packed_20261008/report.md): 고정 후보와 사전 이중 품질 조건을 적용했다. 같은 valid20 탐색이며 독립 일반화 검증이 아니다. 원본 base를 변경하지 않았다. 공식 평가는 각 이미지 4 IoU 평균 후 20장 평균, absent/empty=0. 원본 영상 재추론과 PNG/GT 독립 재채점 일치를 확인했다. 모든 클래스·이미지별 손실과 기존 original349 비교는 comparison.json에 포함한다. raw 단회 시간은 속도 개선 근거가 아니다. - 실제 GPU 복원 probe: {'restoration_checks': [{'source': 'restart59_aspp768_seed456', 'all_parameters_and_buffers_exact': True, 'state_entries': 375, 'same_model_object': True, 'grad_reset': True, 'eval_reset': True}, {'source': 'restart61_pairwise_margin', 'all_parameters_and_buffers_exact': True, 'state_entries': 375, 'same_model_object': True, 'grad_reset': True, 'eval_reset': True}, {'source': 'restart59_aspp768_seed456', 'all_parameters_and_buffers_exact': True, 'state_entries': 375, 'same_model_obj
- [completion.json](experiments/model349_compression/base12_resident_packed_20261008/completion.json): diagnostics_completed=True; base_and_weights_unchanged=True; time_candidate=packed_fast12; independent_generalization_verified=False; new_training_started=False; storage_schema_failure_recovered=True
- [failure_storage_audit_1.json](experiments/model349_compression/base12_resident_packed_20261008/failure_storage_audit_1.json): error=KeyError('model'); complete=False

### base12_singleworker_20261008

- 작업 폴더: [base12_singleworker_20261008](experiments/model349_compression/base12_singleworker_20261008/)
- [cancelled_base_change.json](experiments/model349_compression/base12_singleworker_20261008/cancelled_base_change.json): reason=user explicitly switched subsequent base to exh14; terminate only owned base12 timing processes; complete=False; partial_results_preserved=True; old_controller_pid=2131668; new_base=Model349_exh14
- [failure_idle_initial.json](experiments/model349_compression/base12_singleworker_20261008/failure_idle_initial.json): error=RuntimeError('GPU7 occupied; no worker started: 6, 5'); complete=False

### exh14_base_search_20261008

- 작업 폴더: [exh14_base_search_20261008](experiments/model349_compression/exh14_base_search_20261008/)
- [사용자 지정 exh14 base 후속 탐색](experiments/model349_compression/exh14_base_search_20261008/README.md): 새 기준: Model349_exh14, 원본 FP32 14개, 공식 valid20 image mIoU80.2312313%, 1,755,428,956바이트. source/weight manifest와 원본 PNG/GT 재채점을 감사했다. base12는 비교용으로 유지한다. - base14_audit.json: 원본 재채점·base12 클래스/이미지 손실·test10 일치율. - previous_base_pointer.json: 변경 전 포인터 원문. 현재 포인터는 ../model349_base.json. - protocol.md/run_search.py: GPU7 단일 기준 → FP16+ZIP 저장14 → 조건부 상태복원/HRAMP/단일worker14 → 각3회 교차 시간 → test10. - finalize.py: 전체 PNG/GT 재채점, 클래스/이미지별손실, 반복 시간, test마스크 비교, checksum 아카이브. - 신규 후보를 실제 채택하는 단계와 사용자가 지정한 base전환을 구분한다. 독립 일반화 미확인이다. 기존GPU3,4의 valid와GPU6 test결과를 수집했다. valid260.51초는2GPU/단회이므로 단일GPU기준과 비교하지 않는다. test정답 없음. 원본성적이 오프라인80.23098%와 가까워도 같은valid선택편향은 사라지지 않는다.
- [cancelled_accuracy_priority.json](experiments/model349_compression/exh14_base_search_20261008/cancelled_accuracy_priority.json): reason=User goal is 81% accuracy; stop speed-only repeats, preserve completed quality results and partial outputs; pgid=2137506; timestamp=1791456931.2111201
- [failure_probe_path_initial.json](experiments/model349_compression/exh14_base_search_20261008/failure_probe_path_initial.json): error=CalledProcessError(1, ['/workspace/venv-komap/bin/python', '-B', '/workspace/model349_compress/exh14_base_search_20261008/probe.py']); complete=False

### exhaustive_20261008

- 작업 폴더: [exhaustive_20261008](experiments/model349_compression/exhaustive_20261008/)
- 로컬 요약 보고서·결정 파일 없음. 실행 완료 여부는 아래 근거와 서버 프로세스로 확인한다.
- [exh_summary.json](experiments/model349_compression/exhaustive_20261008/exh_summary.json)

### exhaustive_review_20261008

- 작업 폴더: [exhaustive_review_20261008](experiments/model349_compression/exhaustive_review_20261008/)
- [80.2% 후보 재검토 — 2026-10-08](experiments/model349_compression/exhaustive_review_20261008/report.md): 저장 확률 재생으로 최고 80.2309769%를 재현했다. 12개 base의 대체 모델로 채택하지 않는다. 14개 가중치로 커지고, 같은 캐시의 base12보다 평균 +0.1058512%p지만 A181 이미지 −0.2646623%p로 사전 최악 손실 0.10%p 기준을 넘는다. 원본 영상부터의 추론, 실제 시간/저장량, 독립 일반화 성능은 이 후보에서 측정하지 않았다. 기존 base/릴리스/가중치를 바꾸지 않았다. 평가 방식은 valid20 각 이미지 4개 클래스 IoU 평균 후 이미지 평균이며 empty union=0이다. replay.py는 저장 확률에서 마스크를 새로 생성하고 GT 교집합/합집합을 세어 FP64로 채점한다. 원본 이미지의 GPU 재추론으로 부르지 않는다. - 최고 구성: 322 fold3·4, 326 fold1·2, 329 fold1·3·4; HR336/347, ResNet34 269. 기본 분기는 제외한다. - 고해상도는 항상 쓰는 restart51에 model109/seed456/pairwise 3개를 더한 총 4개다. “후보 3개”를 총 가중치 3개로 해석하면 안 된다. EMA 구간은 restart51로 보낸다. - separate pair flag=1이지만 캐시 메타데이터 20장 모두 pair_weight=0이다. 별도 pair prediction의 효과를 평가하는 실험이 아니다. flag=0에서 0.000025%p 차이는 정규화 경로의 반올림 차이다. route의 pairwise checkp

### generalization_20261008

- 작업 폴더: [generalization_20261008](experiments/model349_compression/generalization_20261008/)
- [12개·10개 FP16 후보의 제한된 일반화 검증](experiments/model349_compression/generalization_20261008/README.md): [한국어 결과](report.md), [사전 고정 범위](protocol.md), [조건부 통계](valid_conditional_stability.json), [데이터/학습 이력 감사](data_exposure_audit.json), [정답 없는 test 출력](test_mask_stability.json). 현재 이미지 100장 중 정답은 90장이고 모든 valid20은 이미 후보 선택에 사용됐다. 이 작업은 독립 일반화 점수를 만들지 않았다. completion.json의 diagnostics_completed는 데이터 감사·조건부 통계·test 마스크 확인의 완료를 뜻하며 independent_generalization_verified는 false다. 데이터 감사에서 train-valid specimen 중복 7개를 확인했다. fold 학습 출력과 동결 배포판의 실제 텐서가 달라 현재 OOF 캐시를 배포 후보의 성적에 사용할 수 없었다. 사전 프로토콜의 학습 노출 가정은 기록된 원래 CV 분할을 적용할 때의 조건부 판단이며, 동결 가중치의 완전한 학습 계보가 인증됐다는 뜻이 아니다. 참조 훈련 코드는 실행 증명이 아니며 파일·텐서 비교 결과는 JSON에 있다.
- [12개·10개 FP16판 일반화 가능성 검증 결과](experiments/model349_compression/generalization_20261008/report.md): **독립 일반화 개선은 두 후보 모두 미확인이다.** 정답이 있는 독립 표본이 없어 현재 가중치의 새 데이터 정확도를 검증할 수 없다. 대신 데이터·학습 이력 감사, 기존 valid의 조건부 재표본화·그룹 제외 분석, 두 후보의 실제 test10 원본 추론을 완료했다. 현재 자료에서는 12개판이 10개판보다 국소 손실과 그룹 민감도가 작다. 12개판은 탐색 후보로 유지하고, 10개판은 기존 이미지 손실 기준 초과로 기각 상태를 유지한다. 기존 기본판·릴리스는 교체하지 않는다. - train 70장/59 specimen, valid 20장/19 specimen, test 10장/10 specimen. 정답은 train/valid에만 있다. - train-valid specimen 중복: A125, A153, A181, A191, A40, A55, A7. 이미지의 A숫자 ID는 기존 group CV 생성 코드의 specimen 정의와 일치한다. - train70과 specimen이 겹치지 않는 valid 부분집합은 13장/12그룹이다. 이 13장도 원래 valid 탐색에 사용됐으므로 독립 holdout이 아니다. 또한 가중치 학습 이력 추적이 불완전해 모든 upstream 학습/선택의 미노출을 인증하지 않는다. - test 중 train/valid와 specimen이 겹치지 않는 것은 6장이다. 나머지 4장은 A167/A51/A87(train), A63(valid)와 그룹을 공유한다. - split 간 동일 디코딩 RGB 이
- [completion.json](experiments/model349_compression/generalization_20261008/completion.json): scope=Exposure audit, selected-valid conditional stability and unlabelled test10 agreement; diagnostics_completed=True; independent_generalization_verified=False; new_training_started=False; fixed_weights_and_configs_unchanged=True; utc_unix=1791436195.1905847

### 초기 경량화·구성 탐색의 개별 채점 기록

아래는 `results/`에 저장된 개별 실험·재추론·속도 측정 기록이다. 관련 디렉터리의 보고서와 함께 확인한다.

- [bundle15_audit.json](experiments/model349_compression/results/bundle15_audit.json)
- [bundle15_test10_completion.json](experiments/model349_compression/results/bundle15_test10_completion.json): model=Model349
- [bundle15_valid20_completion.json](experiments/model349_compression/results/bundle15_valid20_completion.json): model=Model349; mean_image_miou=0.8006435964878685
- [bundle15_valid20_metrics.json](experiments/model349_compression/results/bundle15_valid20_metrics.json): count=20; image_miou=0.8006435964878685
- [compression_frontier.json](experiments/model349_compression/results/compression_frontier.json)
- [eval1.json](experiments/model349_compression/results/eval1.json)
- [eval2.json](experiments/model349_compression/results/eval2.json)
- [eval3.json](experiments/model349_compression/results/eval3.json)
- [eval_l14.json](experiments/model349_compression/results/eval_l14.json)
- [eval_l14b.json](experiments/model349_compression/results/eval_l14b.json)
- [eval_single.json](experiments/model349_compression/results/eval_single.json)
- [force_restart51_valid20_metrics.json](experiments/model349_compression/results/force_restart51_valid20_metrics.json): count=20; image_miou=0.8001889779395832
- [high_source_prune_test10.json](experiments/model349_compression/results/high_source_prune_test10.json)
- [high_source_prune_valid20.json](experiments/model349_compression/results/high_source_prune_valid20.json)
- [lite13_valid20_completion.json](experiments/model349_compression/results/lite13_valid20_completion.json): model=Model349; mean_image_miou=0.8005724554145223
- [lite13_valid20_metrics.json](experiments/model349_compression/results/lite13_valid20_metrics.json): count=20; image_miou=0.8005724554145223
- [lite14_train70_metrics.json](experiments/model349_compression/results/lite14_train70_metrics.json): count=70; image_miou=0.8297679278324429
- [lite14_valid20_completion.json](experiments/model349_compression/results/lite14_valid20_completion.json): model=Model349; mean_image_miou=0.8006503985745343
- [lite14_valid20_metrics.json](experiments/model349_compression/results/lite14_valid20_metrics.json): count=20; image_miou=0.8006503985745343
- [lite_valid_completion.json](experiments/model349_compression/results/lite_valid_completion.json): model=Model349; mean_image_miou=0.8006429602564389
- [lite_valid_metrics.json](experiments/model349_compression/results/lite_valid_metrics.json): count=20; image_miou=0.8006429602564389
- [rep_free_valid20_completion.json](experiments/model349_compression/results/rep_free_valid20_completion.json): model=Model349; mean_image_miou=0.8006447186468693
- [rep_free_valid20_metrics.json](experiments/model349_compression/results/rep_free_valid20_metrics.json): count=20; image_miou=0.8006447186468693
- [timing_269.json](experiments/model349_compression/results/timing_269.json)
- [timing_322.json](experiments/model349_compression/results/timing_322.json)
- [timing_322soup.json](experiments/model349_compression/results/timing_322soup.json)
- [timing_326.json](experiments/model349_compression/results/timing_326.json)
- [timing_329.json](experiments/model349_compression/results/timing_329.json)
- [timing_336.json](experiments/model349_compression/results/timing_336.json)
- [timing_347.json](experiments/model349_compression/results/timing_347.json)
- [timing_base.json](experiments/model349_compression/results/timing_base.json)

### 학습 재현성·환경 차이 기록

- [팀 원본349 100시드 평균 기록](experiments/model349_compression/team_model349_original_seed100_20261009.json): 사용자 보고 80.255658%; 독립 채점하지 않았다.
- [10시드 재학습 비교 프로토콜](experiments/model349_compression/training_reproducibility_seed10_protocol_20261009.json): 설계 기록이며 이 프로토콜의 학습을 실행한 것으로 표시하지 않는다. Claude의 OOF90 작업을 중복 구현하지 않는다.

## 팀 fusion805 재학습 보고 — 사용자 제공, 독립 채점 전

- 2026-10-09 스크린샷: R(exh14) 80.051924%, 1단계 R+E 80.236507%, 2단계 R+E+F 80.267268% (valid20).
- 우리 고정 가중치 fresh 1단계 80.474424% 대비 −0.237917%p. 이 차이는 R 기준 하락 −0.179062%p와 결합 이득 감소 −0.058854%p로 분해된다.
- 팀의 E 결합 이득은 +0.184583%p로 유지됐다. 현재 점수 차이의 약75%는 R 기준 차이에 해당한다. 원인·클래스별 오차 위치는 로그·이미지별 출력이 필요하다.
- 실행 명령·ZIP/커밋·시드·args·학습 로그·이미지별 CSV를 요청했다. GPU 원인 및 재학습의 평균 성능은 아직 확정하지 않는다.
- 근거: `experiments/model349_compression/team_fusion805_retraining_report_20261009.json`.


## Codex HR 추가 학습 valid20 — 실행 전 등록

- ID: `accuracy81_HR_warmstart_valid20_20261009`. 담당 Codex, GPU3 단독.
- 기존 seed1 4fold 추가 학습 가중치만 valid20에 flip4 추론. 새 학습 없음. 비율·덮어쓰기 규칙 고정. Claude B4gray 및 OOF 설계와 중복 없음.
- 서버 경로 `/workspace/model349_compress/accuracy81_HR_warmstart_valid20_20261009/`. 상태: 실행 중. 네 fold 추론·CPU 채점 완료(exit0). 기존 80.474424%, 새 후보 80.482580%, +0.008156%p. 원본 입력 SHA 보존. GPU3 사용 종료.
- 목적: 고정 E HRseed1 교체 후 R+E valid20 비교. valid 반복 사용의 선택 편향은 유지됨.

- Codex valid20 채점 코드 `accuracy81_HR_warmstart_valid20_20261009/compare.py` 준비. 기존 fresh 80.4744236% 재현을 강제하고 같은 비율·마스크 규칙으로 비교. fold별 FP16 저장 후 평균하므로 최종 채택 전 FP32 fold 평균 fresh 추론이 필요함을 기록한다.

- Codex valid20 자동 후속 채점 등록: `finalize.py`, SSH session87648. controller PID2325188의 실제 `/proc` 명령을 확인하며 대기하고, 네 fold 완료 후 `compare.py`를 한 번 실행한다. 학습·추론 재실행 없음.

- Codex 고정 E의 train70 조건부 시편 bootstrap(59그룹, 10,000회, seed819349): 평균 +0.075046%p, 95% 구간 [+0.001772, +0.149893]%p. 레시피 선택 편향을 제거하지 않는 진단이다. 근거 `accuracy81_HR_warmstart_E_oof_20261009/conditional_uncertainty.json`.

- Codex valid20 결과 수집 및 저장 IoU 재합산 확인. +0.008156%p로 81% 미달. fold별 FP16 평균의 스크리닝 결과이므로 기본 릴리스 교체하지 않음. 근거 `accuracy81_HR_warmstart_valid20_20261009/comparison.json`, `report.md`, `decision.json`.

## Codex HR 학습 길이 확대 — 실행 전 등록

- ID `accuracy81_HR_long_fold1_20261009`, 담당 Codex, 물리 GPU3. 실제 유휴 확인(6MiB, 0%).
- 부모는 기존 fold1 원본RGB 4epoch 대조군. 여기서 12epoch 추가, seed819350, 새 AdamW LR3e-5/encoder0.2, FP32 batch1acc8, BN 통계 고정, 마지막 가중치만 평가.
- train70 내부 train52/held18만 사용. 공식 valid/test 미사용. 기존 4epoch 반복 학습 없음.
- 목적: 4epoch의 valid20 +0.008156%p에 그친 효과를 더 긴 학습으로 확장 가능한지 보류 fold에서 확인. Claude B4gray·OOF 구현과 중복 없음.
- 서버 `/workspace/model349_compress/accuracy81_HR_long_fold1_20261009/`, 로컬 같은 ID. 상태 학습·평가·독립 채점 완료. 4epoch 77.691822%, 추가12epoch 77.630001%, −0.061821%p. Ni +0.195292%p, Primary −0.329046%p. GPU3 사용 종료. 12epoch 완료 뒤 flip4 보류 평가, 독립 채점 후 다른 fold 확대 여부 결정.

- Codex 긴 학습 완료 후 자동 독립 비교 코드 준비: 직전 4epoch fold1 보류18 FP16 캐시와 같은 추론 설정·GT로 비교. GPU 학습을 중복 실행하지 않으며 공식 valid/test 미사용.

## Codex 긴 학습의 고정 E 효과 — 실행 전 등록

- ID `accuracy81_HR_long_E_fold1_20261009`, CPU 전용, 보류18 저장 확률. 4epoch HRseed1과 추가12epoch HRseed1만 교체, E 비율 고정. 공식 valid/test 미사용.
- 단독 평균 하락에도 Ni 향상이 있어 E 효과 확인 후 확대 여부 결정. 학습·추론 재실행 없음. 서버 같은 ID. 상태 준비.

- Codex 긴 학습 고정 E CPU 채점 완료(exit0): 78.100906 → 78.094144%, −0.006762%p. HR 단독과 E 모두 손해라 다른 fold의 추가12epoch 확대 미채택. `accuracy81_HR_long_E_fold1_20261009/result.json`, `decision.json`, `report.md`.

## Codex HR soft IoU 손실 — 학습·독립 채점 완료

- ID `accuracy81_HR_softiou_fold1_20261009`, 담당 Codex. GPU3는 현재 타 프로젝트 WavLM 학습 사용 중(NVML PID2495573), 중단하지 않음. CPU 대기 실행기가 실제 유휴 확인 뒤 기존 GPU 잠금·외부 프로세스 감시 적용해 한 번 시작.
- 기존 HRseed1 fold1 원본 가중치에서 4epoch, seed819349, RGB·FP32 batch1acc8·BN고정·LR3e-5 동일. Dice 항만 soft IoU로 변경, CE·경계 손실 유지. 완료된 4epoch 대조군과 비교.
- train70 train52/held18만 사용. 마지막 가중치 고정, 공식 valid/test 미사용. Claude B4gray·OOF와 중복 없음.
- 서버 `/workspace/model349_compress/accuracy81_HR_softiou_fold1_20261009/`. GPU3 유휴 확인 후 기존 실행기가 학습 시작. 목적: 지표와 가까운 손실로 held 성능·E Ni/Al 판단 개선 확인.

- Codex 대기 PID2339477이 기존 controller PID2342415, 학습 PID2342442를 한 번 실행한 것을 실제 확인. 1/4 epoch, 50/624 step, loss=1.0942. CPU 후속 채점 PID2339588도 생존 확인. 완료 후 동일 4 epoch 대조군과 독립 FP16 채점·한국어 보고서 생성 예정. 아직 성능 점수 없음.

- 최종 결과: held18 Dice 대조군 77.691822% → soft IoU 77.666527%, −0.025295%p. Al +0.001243, Al3Ni +0.008676, Primary Si −0.072176, Eutectic Si −0.038923%p. 학습·추론·독립 채점 완료 파일 확인 및 로컬 결과 수집 완료. 전체 평균 손해로 다른 fold 확대 보류; 공식 valid20/test 미사용. 다음은 저장된 확률로 고정 E 영향을 CPU에서 확인.

## Codex soft IoU 고정 E — 실행 전 등록

- ID `accuracy81_HR_softiou_E_fold1_20261009`. CPU 저장 확률 채점만 수행. held18의 B4 0.5 + HR seed1 0.25 + HR seed2 0.25를 고정하고 HR seed1 Dice 4 epoch 대조군을 soft IoU 4 epoch로 교체. 학습·추론 추가 실행 없음, 공식 valid20/test 미사용. 상태 준비.

- 완료(exit0), 입력 SHA 불변 확인: E 78.100906% → 78.100519%, −0.000388%p. HR 단독과 고정 E 모두 개선 없음. soft IoU의 다른 fold 확대 미채택. 로컬 `result.json` 수집 완료. 81% 미달성.

## Codex HR 학습률 1e-4 — 실행 전 등록

- ID `accuracy81_HR_lr1e4_fold1_20261009`, GPU3. 원본 HR seed1 fold1 부모에서 원본 CE/Dice·경계 손실, RGB·BN 고정·FP32 batch1acc8·seed819349·4 epoch·마지막 가중치 고정. 기존 3e-5 대조군 대비 학습률만 1e-4로 변경. train52/held18, 공식 valid20/test 미사용. Claude OOF90/B4gray와 별도. 유휴 확인·잠금·중복 거부 적용. 완료 후 독립 채점 및 고정 E 확인 예정.

- 학습·추론·독립 채점 완료. held18 77.691822% → 77.613944%, −0.077878%p. 클래스 변화 Al +0.032058, Al3Ni +0.142594, Primary Si −0.524656, Eutectic Si +0.038490%p. 메타데이터 동일·입력 SHA 불변 확인. 결과 로컬 수집 완료. 단독 손해로 다른 fold 확대 보류, Ni 개선에 따른 고정 E 효과를 다음 CPU 단계에서 확인.

## Codex LR1e-4 고정 E — 실행 전 등록

- ID `accuracy81_HR_lr1e4_E_fold1_20261010`. CPU 저장 확률 채점, held18, B4/HR seed1/HR seed2 비율 0.5/0.25/0.25 고정. HR seed1의 LR3e-5 대조군만 LR1e-4 후보로 교체. 공식 valid20/test 미사용, 추가 학습·추론 없음. 상태 준비.

- 완료(exit0), 입력 SHA 불변 확인. 고정 E 78.100906% → 78.122005%, +0.021099%p. HR 단독 −0.077878%p와 달리 앙상블에서는 작은 이득. 단일 fold 사후 개발 결과이며 81% 또는 일반화 개선 증거가 아니다. 로컬 결과 수집 완료. 확대 판단 전 시편별 이득·클래스 변화 확인 예정.

- 분석 완료: 10장 개선·8장 악화. E 클래스 변화 Al +0.010712, Ni +0.044222, Primary Si −0.040668, Eut +0.070129%p. 시편 동일 가중 평균 +0.011801%p, 사후 조건부 bootstrap 95% [−0.018055,+0.041400]%p(이미지 평균의 구간 아님, 선택 편향 미제거). 이득이 작고 HR Primary Si 하락이 커 다른 fold 확대 및 릴리스 채택 미실시. `diagnostic.json`, `decision.json`, 한국어 보고서 저장.

## Codex HR seed2 추가 학습 — 실행 전 등록

- ID `accuracy81_HR_seed2_warmstart_fold1_20261010`, GPU3. 기존 HR seed2 fold1 부모와 seed2 전용 분할을 사용. seed1에서 유효했던 RGB·CE/Dice·경계 손실·BN 고정·FP32 batch1acc8·LR3e-5/encoder0.2·seed819349·4 epoch 레시피를 고정 적용. 마지막 가중치만 held fold1 평가. 공식 valid20/test 미사용, Claude OOF90와 별도. 중복 검색상 기존 seed2 추가 학습 없음. 원본 seed2 OOF와 독립 채점 및 고정 E 비교 예정.

- 완료: 원본 seed2 held18 77.323791% → 추가 학습 78.121544%, +0.797753%p. 클래스 모두 개선: Al +0.088049, Ni +0.939257, Primary Si +0.624527, Eut +1.539178%p. 입력 불변·추론 메타 동일 확인 및 로컬 결과 수집 완료. 보고서의 대조군 표기 '4epoch'는 원본 seed2를 뜻하므로 정정. 단일 fold 결과, 공식 valid/test 성능 아님. 다음 고정 E CPU 비교 후 다른 fold 확대 판단.

## Codex HR seed2 고정 E — 실행 전 등록

- ID `accuracy81_HR_seed2_E_fold1_20261010`. CPU held18 저장 확률 채점. 원본 E 0.5 B4 + 0.25 HRseed1 + 0.25 HRseed2에서 seed2만 추가 학습으로 교체. 후속으로 seed1 추가 학습을 고정한 상태에서도 seed2 교체 효과 확인. 공식 valid20/test 미사용. 기존 학습/추론 중복 없음.

- 두 CPU 채점 완료(exit0), 입력 SHA 불변: 원본 E 78.074303% → seed2만 추가 78.158398%, +0.084095%p. seed1만 추가 E 78.100906% → 두 시드 추가 E 78.142796%, +0.041889%p. 단독 개선이 앙상블에서도 유지되나 두 시드 동시 추가는 seed2만 추가보다 낮음. 단일 fold 결과, 조합의 최종 선택은 다른 fold 검증 후 진행. 결과 두 JSON 로컬 수집 완료.

## Codex HR seed2 fold2 추가 학습 — 실행 전 등록

- ID `accuracy81_HR_seed2_warmstart_fold2_20261010`, GPU3。fold1과 같은 4 epoch·LR3e-5·seed819349·RGB·CE/Dice/경계·BN 고정·FP32 batch1acc8 레시피를 원본 seed2 fold2 부모와 전용 분할에 적용. 마지막 가중치 고정 평가, 공식 valid20/test 미사용. 기존 seed2 fold1 완료 확인 후 별도 디렉터리 실행. 시편 분리·입력 SHA·GPU 유휴·잠금·중복 거부 적용. 독립 채점 후 고정 E에서 재검증 예정.

- 환경이 네트워크 제한으로 변경되어 SSH가 `Operation not permitted`로 거부됨. 마지막 확인은 waiter PID2352767/controller2352770/학습2352815 실행, 독립 채점 SSH 세션18521 시작까지. 현재 서버 생존·완료 여부는 확인 불가이며 중단으로 간주하지 않음. 연결 복구 후 같은 실행의 completion/finalizer 파일부터 확인하고 재실행 금지.

- Codex 재개 확인(2026-10-10): SSH 복구. fold2 기존 PID와 active_child PID2354368 모두 종료, completion/finalizer_completion 및 train/eval exit0 확인. 독립 채점 77.899001 → 78.692490%, +0.793489%p. 입력 SHA 불변·메타 동일, 실패 파일 없음. 결과·로그 로컬 수집 완료. 재학습/재추론 없음.

## Codex HR seed2 고정 E fold2 — 실행 전 등록

- ID `accuracy81_HR_seed2_E_fold2_20261010`, 담당 Codex, CPU 전용. 서버 같은 ID. held18 저장 확률, 원본 E 및 seed1 추가학습 E에서 seed2만 교체 비교. 비율 0.5/0.25/0.25 고정, 공식 valid/test 미사용. 상태 준비. seed1 대조군은 `accuracy81_HR_warmstart_fold2_20261009/original_eval`, seed2 후보는 완료된 fold2 실험. Claude 기록·원본·릴리스 보존.

- fold2 고정 E 두 CPU 채점 완료(exit0), 입력 SHA 불변. 원본 E 79.143088 → seed2 교체 79.189795%, +0.046706%p. seed1 추가 E 79.234603 → 양쪽 추가 79.254279%, +0.019675%p. 결과 로컬 수집 완료.

## Codex HR seed2 fold3·4 확대 — 실행 전 등록

- ID `accuracy81_HR_seed2_warmstart_fold3_20261010`, `accuracy81_HR_seed2_warmstart_fold4_20261010`. 담당 Codex, 물리 GPU3 순차 사용. 상태 준비. 서버 `/workspace/model349_compress/` 아래 동일 ID.
- fold1·2 단독 및 고정 E 개선을 확인하여 같은 고정 4epoch 레시피를 각 seed2 부모·전용 분할에 적용. 마지막 가중치만 held fold 평가, 공식 valid/test 미사용. GPU 실제 유휴·전역 잠금·외부 프로세스 감시·시작 파일 중복 거부 유지. Claude OOF90/B4 설계와 중복 없음. 완료 후 독립 FP16 채점·고정 E 비교 및 전체 valid20 후보 검증.

## Codex HR seed2 후속 자동 검증 — 실행 전 등록

- CPU ID `accuracy81_HR_seed2_E_fold3_20261010`, `accuracy81_HR_seed2_E_fold4_20261010`, `accuracy81_HR_seed2_followup_20261010`. 기존 fold3→4 순차 실행(PID2389291) 완료 파일과 실제 /proc를 대조하며 대기, 독립 채점 성공 후 저장 확률 고정 E 비교·held70 합산. 중복 학습/추론 없음.
- GPU ID `accuracy81_HR_seed2_valid20_20261010`, 물리 GPU3. 준비 상태이며 held70 고정 E 개선>0 및 3개 이상 fold 개선 시에만 원본 B4/HRseed1을 고정하고 HRseed2 4fold만 교체한 후보의 valid20 추론. FP32 각 fold 확률을 평균한 뒤 가족 평균에만 FP16 1회 반올림(기존 fresh 규칙). 원본 R/B4/HRseed1 fresh 확률 재사용, seed2는 새 가중치 원본 영상 추론. 전체 1단계 마스크 독립 PNG/GT 채점 및 입력 SHA 불변 확인. 공식 valid 반복 개발 한계 유지, 자동 릴리스 교체 없음. Claude B4ws 결합은 이번 고정 후보에 포함하지 않음.

- 실제 실행 확인 **2026-10-10 09:58:55 KST**: 순차 controller shell PID2389291, fold3 waiter2389292/controller2389295/train2389321. fold3 1/4epoch 300/636 step, GPU3 4054MiB/75% 사용 확인(직전 조회). fold4는 앞 단계 성공 후 실행.
- 후속 CPU supervisor **PID2389701** 생존, `status.json` phase=`waiting_existing_sequence`, fold3 대기. fold3·4 완료 후 고정 E CPU 비교 → held70 gate → 조건 통과 시 고정 seed2 교체 전체 valid20 후보 FP32 평균 추론·PNG 독립 채점으로 자동 연결. `/workspace/model349_compress/accuracy81_HR_seed2_followup_20261010/followup.log`, `held70_gate.json`, `completion.json`, `failure.json`을 확인한다. valid20 출력은 `/workspace/model349_compress/accuracy81_HR_seed2_valid20_20261010/`. 아직 fold3/4·valid20 결과 없음, 81% 목표 미달성 상태 유지.
- valid20 사전 대조: 원본 B4/HRseed1/HRseed2 저장 확률로 만든 E와 기존 fresh E의 라벨 20장 0픽셀 차이 확인(`preflight.json`). fold2 report의 부모를 4epoch라 부르던 복사 잔재 정정, 원본 protocol의 held fold1 오표기는 감사 메모로 명시하고 실행 메타 보존.

### Codex fusion805 경량화 — 실행 전 등록 (2026-10-10)

- 실험 ID: `fusion805_compression_20261010`; 상태: 준비 중.
- 사용자 요청: 349→lite14와 같은 방식으로 fusion805 가중치 수 축소.
- 범위: 기존 R 및 결합 규칙 고정, E의 B4/HRNet 가족 제거 고정 6구성 비교. Claude의 신규 구성원 추가·B4ws·gamma 재선택과 별개.
- 데이터: 기존 canonical valid20 부모 확률 재생; test 미사용. 반복 valid 탐색이며 일반화 검증 아님.
- 서버: `/workspace/model349_compress/fusion805_compression_20261010`; CPU 2 threads, GPU 미사용. 실제 ps/nvidia-smi 확인: Claude gamma CPU 및 GPU4 패키지 작업 생존, 조작하지 않음.
- 결과 전 기준: 평균 손실≤0.01%p, 최악 영상 손실≤0.10%p. F는 각 축소 E 입력에서 다시 계산, 원래 F 출력 재사용하지 않음. 1·2단계 별도 비교. 원래 릴리스/ZIP 불변.

### Codex fusion805 单fold 제거 — 실행 전 등록

- 실험 ID: `fusion805_compression_20261010/fold_prune`; 상태: 준비 중.
- 가족 제거 6개는 평균/영상별 동시 gate 미통과. 가장 가까운 seed2 전체 제거(23개)는80.501714%, 최악−0.122002%p.
- 후속 고정 후보: seed2 fold4 한 개만 제거(27→26). B4·seed1·R·F·family 비율 고정, seed2 나머지3fold 평균. fold 선택 탐색 없음.
- 원본valid20 HR seed2 3fold 추론(정답 미사용), 두 입력 묶음 GPU6·7에서 별도 출력. 서버 실제 GPU 점유 및 기존 잠금 재확인 후 실행. Claude GPU4 및 gamma CPU 작업과 겹치지 않음. test 미사용.

## Codex HR seed2 fold3 중단 복구 — 실행 전 등록

- 2026-10-10 10:40 KST: 기존 fold3·순차 실행·후속 supervisor 실제 PID 모두 없음. fold3 controller_failure는 외부 GPU3 프로세스 유입에 따른 자기 child 종료를 기록. GPU3 NVML PID3137593(3328MiB)이 실행 중이며 건드리지 않는다. fold4/E-fold3/E-fold4/valid20 미시작 확인.
- 새 ID `accuracy81_HR_seed2_warmstart_fold3_resume1_20261010`, 담당 Codex, GPU3 유휴 대기. 원본 fold3 중단 디렉터리는 보존. train_state.pt epoch1 및 model/optimizer/scheduler/scaler/모든 RNG 상태 확인. 별도 디렉터리에 동일 SHA 사본을 만들고 `--resume`로 epoch2부터 총4epoch를 이어간다. 중단 epoch2의 미저장 부분은 복구할 수 없어 해당 epoch만 다시 수행; epoch1 재학습 없음. 부모·분할·레시피 동일, 상태파일 args 엄격 대조 유지.
- 새 orchestration ID `accuracy81_HR_seed2_recovery_followup_20261010`: resume3 → 기존 미시작 fold4 → 독립채점·고정E·held70 gate → 조건 통과 시 기존 미시작 valid20. 기존 실패기록 보존; E-fold3와 valid20의 fold3 가중치 경로만 새 resume 디렉터리로 변경. 상태 준비.

- Codex `fusion805_compression_20261010/fold_prune` 실행 확인: GPU6 PID2393573, GPU7 PID2393572, 각10영상 HR seed2 4fold 확률 생성 중(6/20 family parity exact). checkpoint SHA 검사 통과, GT 미사용. 완료 후 fold4 제거 CPU재생·독립 PNG채점은 대기 실행기에서 자동 순차 실행. 기존 F 재사용 출력이 아닌 새 입력 재계산.

- 복구 실행 실제 확인 **2026-10-10 10:42:12 KST**: recovery supervisor PID2394992와 GPU 대기 PID2395056 생존. GPU3 외부 NVML PID3137593(3334MiB/58%) 때문에 대기, 학습 재개 아직 미시작. 원본 입력 SHA 불변, epoch1 checkpoint 및 args 사본 SHA 동일 확인(`resume_receipt.json`), 실패 로그 로컬 수집. 원본 epoch1·실패 디렉터리 보존. 새 상태 `/workspace/model349_compress/accuracy81_HR_seed2_recovery_followup_20261010/status.json`, 로그 `followup.log`, 완료 `completion.json`, 오류 `failure.json`. 후속 E/valid20의 fold3 경로를 재개 폴더로 연결 완료. 81% 목표 active, 새 성능 점수 없음.

- 실제 재조회 **2026-10-10 10:43:26 KST**: recovery2394992와 waiter2395056 생존, 외부 GPU3 PID3137593 3328MiB 사용. fold3 재개·fold4·valid20 미시작, 재실행 없음.
- 완료된 fold1·2 보류36/30시편 CPU 합산 진단: seed2 HR +0.795621%p, 고정 E +0.065401%p(20개선/16악화), E 네 클래스 모두 평균 개선. 시편 하나 제외 시 이미지 평균 이득 범위 [+0.029385,+0.091318]%p. 시편 동일가중 조건부 bootstrap95% [−0.034550,+0.225836]%p는 0 포함. 양 시드 추가 E에서 seed2 교체 효과 +0.030782%p. 부분 결과의 사후 진단이고 81%·전체 valid·독립 일반화 검증 아님. 근거 `accuracy81_HR_seed2_recovery_followup_20261010/held36_diagnostic.json`. 기존 확대 gate 유지, 다른 fold와 전체 후보 검증 필요.

- 단일fold 후속의 사전 범위 확장(점수 생성 전): seed2 유지fold를 [1,2,3], [1,2], [1] 세 고정 prefix로 비교한다. 각각2단계26/25/24개. 남기는fold를 valid로 순열 탐색하지 않으며, 같은 원본4fold추론을 재사용한다. gate 및 시편제외 선택 그대로.

## Codex GPU0 이동 — 사용자 지시, 실행 전 등록

- 사용자가 GPU0의 메모리만 있고 사용률0이면 GPU0 사용을 명시 허용했다(기존 GPU3 전용 제한의 이번 작업 예외). GPU0 UUID `GPU-cc2e4f3a-b098-db06-d993-3c46e4fa8b78`, 16GiB 중260MiB, 사용률0, 기존 NVML PID2123683 254MiB 확인. 기존 프로세스는 종료하지 않는다. 연속3회 유휴·상주메모리 동일 확인 후 우리 GPU3 대기 waiter만 종료한다.
- 새 ID `accuracy81_HR_seed2_warmstart_fold3_gpu0_20261010`, 담당 Codex, 물리GPU0. epoch1 체크포인트·optimizer/scheduler/RNG를 별도 사본으로 보존하고 epoch2부터 총4epoch 재개. GPU3 중단/대기 기록 보존.
- 새 supervisor `accuracy81_HR_seed2_gpu0_followup_20261010`: GPU0 resume3 → 미시작 fold4(GPU0) → CPU 독립채점·고정E·held70 → 조건 통과 시 valid20(GPU0). 기존 상주 PID는 메모리 baseline으로 감시하고 새 외부 PID 또는 상주 메모리128MiB 초과 증가 시 우리 child만 종료한다. GPU0 전용 잠금 적용. 상태 준비.

- GPU0 이동 실제 확인 **2026-10-10 10:53:29 KST**: supervisor2396504 / 학습child2396619(NVML3180468). 로그 `resumed from epoch 1; next epoch=2` 확인, train53/636 steps, width1278. GPU0 4814MiB/89%, 기존 상주 PID2123683 254MiB 유지. GPU3 waiter2395056와 supervisor2394992는 우리 waiter SIGTERM에 따라 종료, 의도된 이동 기록과 실패 이력 보존. 사본 SHA 동일·구문 검사 통과, 원본 입력/학습 epoch1 보존. 이후 fold4·valid20도 GPU0 전용 잠금과 baseline 감시 적용. 새 상태 `/workspace/model349_compress/accuracy81_HR_seed2_gpu0_followup_20261010/`, 결과 미확정·81% active.

### Codex fusion805 HR seed2 부분집합 — 실행 전 등록

- 세 prefix24/25/26후보는모두gate미통과. 원본seed2 4fold 캐시20/20 exact 및가중치SHA보존확인.
- 추가 범위: seed2 비어 있지 않은 전체부분집합14개(크기1~3)와control. 동일저장fold맵재사용,추가GPU작업없음. CPU2threads,subset_search 새디렉터리. 반복valid 탐색이며 gate/시편제외내부 전체재선택/부모선택편향한계기록.

### Codex fusion805 경량화 완료

- 상태: 완료. 가족제거6개, prefix3개, seed2 전체부분집합14개 비교. GPU6·7 HR seed2 원본추론 종료, 후속CPU 종료.
- 선택: seed2 fold2·3 제거/fold1·4 유지. R14+B4 4개+HR seed1 4개+HR seed2 2개+F1 =25개.
- 2단계80.5006890946→80.5158924815%(+0.0152034%p), 최악A181−0.071187%p/사전gate통과. 용량4.554→3.936GB(−13.568956%).
- 1단계26→24개,80.4746206171→80.4888441884%(+0.0142236%p),최악−0.072857%p/gate통과.
- subset조건부LOSO:1단계80.4831598367%,2단계80.5102186066%. 부모선택및반복valid편향남음,독립일반화검증아님.
- HR seed2 4fold 원본20장 mean캐시확률exact; 다른부모는캐시재사용. F재계산/선택PNG20 별도공식채점/IoU와gate및LOSO독립재집계/prefix중복구성IoUexact/문법CLI검사/ZIP무결성통과.
- 새artifact `Model349_fusion805_lite25_exploratory_code.zip`(가중치·데이터제외),`experiments/model349_compression/fusion805_compression_20261010/report.md`. 원래릴리스와ZIP불변/test미사용/전체fresh시간·속도·재학습성능미검증.

- GPU0 fold3 재개 학습 완료 2026-10-10 11:07:35 KST: 총4epoch 마지막636/636, train receipt exit0. 학습child2396619 종료, 자동 보류17 flip4 평가child2398971 실제 생존 확인. supervisor2396504 유지. train 로그·receipt 로컬 수집. 아직 독립 채점 점수 없음, loss로 개선 판단하지 않음.

### Codex fusion805 추가 경량화 round2 — 실행 전 등록

- 사용자 목표: 성능 유지하며25개에서더축소. ID `fusion805_compression_round2_20261010`, 상태 준비. 원래27개 및lite25둘모두평균손실≤0.01%p/최악영상손실≤0.10%p 유지.
- 첫고정범위: seed1 전체제거/seed2 비어있지않은15부분집합, B4 .5 + seed2 .5. checkpoint20~23개. R14·F·규칙고정, F새확률재계산. 과거seed2원본4fold맵 재사용/새부모GPU추론없음.
- GPU7 F재생가속(원래27/lite25 마스크20장exact gate). 실제nvidia/ps/sibling잠금확인후실행. 실패시CPU별도attempt로보존. Claude GPU4/CPUgamma및Codex GPU0 추가학습과별개.
- 기존결과/ZIP유지/test미사용/반복valid후속탐색. GPUF시간은모델전체추론시간이아님. 첫단계불충분하면HRseed1fold맵생성후시드동시부분제거확대.

- GPU0 fold3 결과 수집 2026-10-10 11:15:43 KST: 학습/평가 exit0·입력 SHA 불변·추론 메타 동일 확인. 독립FP16 HR 83.131005687 → 83.075483440%, -0.055522247%p. 클래스 전부 하락. 고정E 83.629062371 → 83.582389800%, -0.046672571%p. seed1추가E에서 seed2교체 -0.056585551%p. 결과 JSON·로그·완료·report 로컬 수집. 보류fold의83%는전체81%달성증거가 아님. supervisor2396504가 fold4 단계로 전환, 동일 고정레시피4epoch GPU0 진행. held70 기존gate유지.

### Codex fusion805 round2 HR seed1 fold맵 — 실행 전 등록

- 20~23개 seed2단독15부분집합 전부 이중gate미통과. 21개keep14 평균80.508303%이나최악−0.161390%p(vs25)로기각. 20개keep4 평균80.518957%이나최악−0.337897%p기각. GPU F재생대조27/25 20장exact,작업종료.
- 다음고정범위: seed1 전체16부분집합×seed2 [없음,1,4,1·4], 양시드없는구성제외. HR총수≤5(전체24개이하) 및25reference,27control. 가족없으면남은HR비율0.5, 양쪽있으면0.25/0.25; B4 .5/R14/F고정.
- seed1 원본4fold 확률생성만GPU6·7(각10영상),seed2기존맵재사용. family20장캐시exact/checkpoint SHA검사/GT미사용. CPU/GPU작업실제확인후시작. Claude GPU4·CPUgamma및GPU0추가학습과별개. 결과별도round2/dual_seed,실행전후R/weights보존.

### Codex fusion805 round2 R Al 경로 제거 — 실행 전 등록

- 병렬CPU캐시실험 r_prune. E lite25 고정/F동일확률재계산. R347 제거(총24),R322 2fold 제거(23),둘제거(22). ratio재선택없이해당경로gamma만0. Claude R B4ws gamma좌표재선택(E′)과다른고정입력/목적.
- 원래R control14 캐시로R재생canonical마스크exact gate, E25 stage1/2 exact gate. 원래27 및25이중손실기준/시편제외재선택. GT없는PNG생성후별도채점. GPU미사용/새부모추론없음. 기존가중치및R설정불변.

- R첫attempt canonical과A11_step2 1pix차이로gate중단보존. `r_prune_v2`로OpenCV2threads/저장cacheR exact검사/canonicalR difference20장기록/canonical25와cache재생25별도비교. 원래27/canonical25/cache25세gate. 실패한r_prune파일덮어쓰지않음.

- R단독3제거미통과(cacheRcontrolexact/canonical과미세차이별도). 양시드후속은점수생성전 R4variant×59E +R-only3+대조3=242설정으로확장. 같은E의F1회계산/R만4가지결합하여GPU추가비용최소화. CPUPNG채점/세기준gate및시편제외전체재선택. valid후속탐색한계유지.

- GPU0 seed2 fold4 학습 완료 2026-10-10 11:32:18 KST: 4epoch/636step 모두 완료, train receipt exit0. trainchild2400305 종료, 자동 held17 flip4 평가child2402845 실제 생존 확인. supervisor2396504 유지. train 로그·receipt 로컬 수집. 독립 점수 및 held70 gate는 아직 미생성, 81% 미달성 유지.

- GPU0 seed2 held70 합산 완료 2026-10-10 11:40:29 KST: fold4 HR +0.389978%p, E +0.060459%p, 두시드E에서 seed2효과 +0.038630%p. held70 HR 79.109933653 → 79.600335044%, E 79.985862008 → 80.022844797%, +0.036983%p. E3/4fold개선, 사전gate통과. seed1추가E에서 seed2효과 +0.011470%p. 70records 로컬독립재합산 일치, 결과·gate 수집.
- 자동 전체valid20 seed2교체 고정후보 시작: waiter2403997/controller2404000/fold1평가2404028 실제 생존. GPU0, FP32 fold확률평균 뒤 가족1회FP16반올림, 기존 R/B4/HRseed1 fresh 재사용. 마지막에는전체20마스크PNG/GT독립채점. 전체점수아직없음/81%미달성/릴리스보존.

### Codex fusion805 lite23 실행 검증 — 실행 전 등록

- 242설정 캐시 비교에서23개(R14/B4 4/HRseed1 2·4/HRseed2 1·4/F1)만23개 이하 평균·최악영상 gate통과. valid20 80.517468151%; 단 전체검색 조건부LOSO 80.470848735%, lite25대비−0.045043747%p로미통과. 기본모델교체없음.
- GPU6·7 각각원본10장 E8체크포인트 fresh inference. 이전검증 R14 raw valid20 재사용하며SHA/입력대조, CPUF 및 독립PNG채점. source/config/checkpoint SHA 동결, GT추론미사용. 새경로 round2/fresh23. 기존GPU0/3/4 작업건드리지않음.

- lite23 선택CPU/GPU stage1/2 20장exact/독립PNG점수80.5174681510/순수Python242설정 gate 및LOSO 재집계 일치. 23개3,317,816,946bytes(원본−27.137913%,lite25−15.699170%). 기본교체없음/LOSO손실기록.
- fresh23 첫 supervisor는구문실패(부모추론미시작), 수정시 다른 Claude B4학습이 GPU6·7을 점유해유휴gate중단(모델미시작). 타작업종료하지않음. waiter2406914가15초간격 실제사용률/메모리 확인→3회유휴후GPU잠금→freshE20→CPUF→독립PNG채점 자동진행. 검증 validation.json 아직없음/전체목표미완료.
- 코드용 Model349_fusion805_lite23_exploratory_code.zip 생성,가중치/데이터제외,ZIP무결성/구문/CLI확인. 원본ZIP및lite25보존. 대기상태는 round2/fresh23/waiting.json.

- 후속goalturn: 이전turn은후보코드/독립재집계/실행queue생성으로progress. 현재waiter2406914 실제ps생존확인, GPU6·7 각각14.238GB/99~100% 사용이라검증대기. 입력·소스 실행전후 SHA 기록 및 R14 이전parent_frozen/weights_unchanged와 대조, 재사용R마스크SHA보존 추가. 아직fresh검증완료아님. artifact소스보강후재생성/ZIP무결성확인.

- fusion805 goal 차단감사: GPU6·7 다른학습점유가연속3goalturn 동일조건으로확인됨. waiter2406914 현재ps생존(4분56초), 두GPU14.238GB/94~100%, Claude fold1학습300/2000 확인. 원본추론미시작이며 validation.json 없음. 다른작업간섭없이진행하려면GPU유휴라는외부상태변화필요. 자동대기/후속검증queue는유지, 모델채택/목표완료는미입증. pending_validation_status.json 기록.

- GPU0 seed2 전체 valid20 완료: 4fold 추론 모두 exit0·입력 SHA 불변, supervisor 완료 및 릴리스 미교체 확인. 고정 R+E 독립 PNG 채점 80.474423601792 → 80.487864161107%, +0.013440559315%p. 20records 로컬 재합산 일치; 10영상 개선/10하락, 클래스 변화 Al +0.012227066/Al3Ni +0.038500278/Primary Si -0.003892840/Eutectic Si +0.006927733%p. report/comparison/완료/로그 로컬 수집. 81% 미달성, 작은 개발 valid 이득이며 독립 일반화 주장하지 않음. GPU0 종료 후260MiB/0% 관측, 기존 상주 작업 보존. 후속 실험은 Claude 공유 기록 확인 후 별도 등록하며 기존 학습·추론 재실행하지 않음.

### Codex seed2 원본·추가학습 혼합 held70 — 실행 전 등록
- ID `accuracy81_HR_seed2_blend_held70_20261010`, CPU 전용. 기존 저장 확률만 사용, alpha=[0,.25,.5,.75,1]의 seed2 원본/추가학습 선형혼합; E B4 .5/HRseed1 .25/HRseed2 .25 고정. 70영상 동일 분할·SHA 확인 및 fold 제외 선택 재평가. 공식 valid/test 미사용, Claude OOF90/B4ws/gamma와 별개. 원본·완료 결과·릴리스 보존.

### Codex fusion805 lite23 최종 검증 GPU0 이동 — 사용자 명시 지시
- 사용자 "최종 검증 gpu0에서 진행해"에 따라 GPU0 사용. 기존GPU0 seed2평가종료/260MiB·0% 및 관련프로세스없음 확인. 기존상주작업보존. GPU6·7대기waiter2406914는우리프로세스만종료하고기록보존.
- 새경로 round2/fresh23_gpu0. GPU0 관련공유lock전부확보/3회유휴확인 후10장씩2shard순차원본E추론→동일F CPU→독립PNG20채점/원본27·lite25·캐시23 gate. R14는이전raw결과SHA대조재사용. 입력/원본가중치/소스불변검사,GT추론미사용. 검증및LOSO한계는그대로기록.

- seed2混合held70 완료: alpha0/.25/.5/.75/1 E 79.985862008/80.004472913/80.014694003/80.021729195/80.022844797%. 최고alpha1은기존전체valid검증후보와동일,혼합으로추가개선없어valid확대기각. fold제외선택 변화 +0.032102945%p,fold3 -0.046672571%p. 입력SHA불변·70records 재합산일치·결과/로그로컬수집. GPU추론중복없음/공식validtest미사용/81%미달성.

### Codex seed2 same-trajectory weight midpoint — 실행 전 등록
- ID `accuracy81_HR_seed2_weights_midpoint_fold1_20261010`. 원본 seed2fold1과완료4epoch의가중치50:50보간, BNbuffers/구조/전처리동일강제, CPU생성·GPU0held18 flip4추론만. 확률혼합과달리단일망파라미터후보,기존상이한부모soup와구분. 추가학습없음/공식validtest미사용. 실제유휴3회·잠금·타사용자보존감시유지. 입력SHA보존/별도출력,완료후독립채점과고정E비교로확대판단.
- GPU0 실제 시작확인: supervisor2409616/추론child2409643, shard0 진행. 이전GPU6·7 waiter2406914 종료확인/cancelled_for_gpu0.json 보존. GPU0공유lock확보/3회유휴gate통과, 새fresh23_gpu0/running.json 기록. 아직점수없음/LOSO한계및기본lite25유지.

- midpoint 첫 시도는 기존 GPU0 공유 잠금으로 추론 전 중단(실패 기록 보존). GPU0 lite23 supervisor2409616/child2409643 사용 확인, 간섭 없음. 새 평가 ID `accuracy81_HR_seed2_weights_midpoint_fold1_20261010_eval1` 별도 출력·CPU 대기 실행기 등록. 생성된 midpoint.pt만 참조, 재생성/재학습 없음. 공유 잠금 해제와 GPU0 유휴 후에만 안전 controller 시작. held18 후 독립 채점은 별도 진행.
- fusion805 GPU0 중간재현검사: supervisor2409616/child2409643 실제생존확인, GPU10.088GB/97%. 완료4영상의 fresh E FP32 확률을동결된 B4/HRfold 재생결과와GT없이독립비교:4/4 확률배열exact/최대차이0/라벨변경0. partial_probability_parity.json,로컬사본보존. 최종20영상PNG채점은아직미완료.
- GPU0후속goalturn 검증: supervisor2409616/child2409643 실제ps생존(12분47초), 5영상완료/오류receipt없음. 5/5 fresh확률 전체exact/라벨차이0 독립대조확인. 기존모델/선택config/실행소스변경없음, 전체20장및최종PNG채점대기. 이번turn은새완료영상재현증거+verified wait.
- fusion805 GPU0 shard0 원본10장 완료: receipt10영상/입력 및 checkpoint SHA 재확인일치. fresh E FP32확률10/10 기존선택캐시재생과exact(라벨변경0). supervisor2409616 정상 유지, 다음child2412699/shard1 실제ps생존 확인. 두번째10장 추론시작/최종20장PNG채점미완료. 첫receipt/중간parity 로컬수집.
- fusion805 GPU0 원본20장 E추론 exit_codes=[0,0] 완료. 20/20 FP32 fresh확률 동결캐시재생 exact. 후속postprocess는stage1부모assets/refiner/epoch8.pt 경로부재로중단(실패receipt보존/모델추론재시작없음). 실제학습완료refiner `/workspace/model349_compress/accuracy81_oof_zgray_20261009/training/epoch8.pt`, SHA932606f8...830926 동일확인후 별도retry.py/로그/PID2415336에서 저장20확률재사용해CPUF·독립채점진행. 원래동결소스/실패로그불변.
- fusion805 lite23 최종 검증 완료: GPU0 E20 원본추론[0,0]/20FP32확률exact, refiner경로수정retry CPUF20·독립PNG20채점 80.51727047204673%. 원본27 +0.016581377%p/최악−0.068943002%p, lite25 +0.001377991%p/최악−0.054365598%p로평균·최악영상 기준통과. 캐시23과7pix차이는모두이전rawR와canonicalR차이로독립확인. 원래동결소스SHA일치/순수Python점수및gate재집계일치. 우리supervisor/추론/retry PIDs모두종료확인; 새GPU0타작업에는간섭없음.
- 23checkpoint/3,317,816,946bytes(원본−27.137913%); lite23 탐색코드ZIP에최종증거포함/무결성및SHA파일갱신. 원래모델/릴리스보존. 기존valid20기준경량화검증완료이며, 조건부LOSO검색−0.045043747%p/독립일반화미검증/속도미측정은report와completion_audit에명시. 기존lite25기본유지.

### 81% 목표 기준 변경 — 사용자 명시 채택
- 사용자가 최종 검증 반영 lite23 코드ZIP을 base로 지정. `ACCURACY81_BASELINE.json`/`.md`에 패키지SHA·구성SHA·최종검증SHA 고정. 기준전체valid20 80.51727047204673%,23가중치/R14+B4 4+HRseed1 2·4+HRseed2 1·4+F1. 이후후보는lite23 전체와비교. 기존27개1단계나held점수로기준개선대체하지않음. 원본/릴리스/lite25/과거LOSO실패기록보존. 81%미달성·반복valid한계유지. 대기midpoint는held진단만유지하고확대시새base에맞춰등록.

### Codex fusion805 추가경량화 round3 — 실행 전 등록
- 사용자 "더 경량화 시도해". 고정범위 HRseed1/2 모든부분집합 총1~4개×R14/13/12/11, 전체checkpoint≤22:578후보+5대조=583설정/164E그룹. 기존round2 seed2 1·4 중심 제한을 해제해2·3 포함. B4/F/교체규칙 고정. GT없는캐시PNG생성 후 독립채점, original27/lite25/raw lite23/canonical lite23 평균손실≤.01%p/최악≤.10%p, 시편제외내부전체재선택. 반복valid한계유지.
- ID fusion805_compression_round3_20261010. GPU7 유휴6MiB/0% 확인 후 관련GPU7공유lock확보·3회유휴gate,2shard순차 F캐시재생. GPU0midpoint/GPU6등타작업불변. 새부모추론없음. 통과후보최종원본영상검증은사용자지정GPU0유휴후. 이전검증된rawR/HRfold cache/가중치SHA 대조·입력보존·대조27/25/23 PNGexact.

- seed2 midpoint held18 완료: 입력불변/추론메타동일, HR77.323790757→77.906942535(+.583151778),E78.074303268→78.111187194(+.036883926)%p. 기존4epoch후보 HR78.121543717/E78.158398296보다낮아보간확대기각. 결과/로그수집.

### Codex seed2 focal fold1 — 실행 전 등록
- ID `accuracy81_HR_seed2_focal_fold1_20261010`, GPU0유휴/잠금검사후원본seed2fold1에서동일4epoch레시피. CE항만focal gamma2로교체(.5focal+.5Dice),경계손실/BN고정/seed819349/LR3e-5/기타조건동일. 완료CE대조군재사용,중복학습없음. held18만평가/공식validtest미사용/Claude OOF90및B4실험별개. 확대시사용자채택lite23 전체stage2와비교,기존원본과릴리스보존.
- round3 실제실행확인: supervisor2416386/child2416394 ps생존, GPU7전용잠금·유휴gate통과. 첫영상 A117_step4_sample2 583설정 생성완료(100.3초),27/25/raw23/canonical23 대조exact gate통과. 후속영상진행/결과미확정. 자동2shard→별도PNG채점 및 조건부LOSO까지run.py 감독. 기존lite23코드ZIP/기준SHA 불변.

- seed2 focal fold1 완료: 학습/평가 exit0·입력SHA불변·추론메타동일·18records로컬재합산일치. HR78.127072575%,동일CE78.121543717% 대비+.005528858%p. 고정E78.138086095%,CE78.158398296% 대비-.020312201%p. 원본대비는HR+.803281818/E+.063782826%p이나대응CE보다E손해여서확대기각. 결과/로그/완료/receipt수집,공식validtest미사용/lite23기준유지/81%미달성.

### Codex lite23 seed2 warm fold1·4 전체 검증 — 실행 전 등록
- ID `accuracy81_lite23_seed2warm14_20261010`, CPU전용. lite23 유지seed2fold1·4만완료4epoch가중치확률로교체,기존valid20 FP32추론재사용/2fold평균후FP16가족반올림. B4/seed1fold2·4/R14/F고정,F는새E로재계산. 원래lite23 E20 확률exact gate통과후전체PNG독립채점. 학습/추론중복없음/타GPU사용없음. 기존seed2 4fold/27개stage1후보와다른lite23전체후보,반복valid한계유지/릴리스보존/81%목표.

- lite23 seed2warm14 전체valid20 완료:80.52331198667409%,기준80.51727047204673% 대비+.006041514627%p. 원래E20 exact/입력SHA불변/F새E재계산/PNG독립채점/20records로컬재합산일치. 최악 -0.089355375%p,시편제외평균 [0.0011557311569064168, 0.011062403545459901]. 결과수집,작은개발이득으로lite23기준교체없음/81%미달성.

### Codex seed2 native crop scale jitter fold1 — 실행 전 등록
- ID `accuracy81_HR_seed2_scalejitter_fold1_20261010`, GPU0. 原본CE/Dice/경계/4epoch/BN고정/seed819349 동일, native crop 크기 .8/1/1.2 균등무작위 후512resize. 별도결정적RNG stream3로crop·기존augmentation RNG보존,확대crop 작은영상reflectpad. 기존Affine .8~1.2는crop후변환이므로native문맥변경과구분. 학습만변경/추론scale2고정/held18만평가/공식validtest미사용. 원본scale추론실험과다름/Claude중복없음. 완료CE대조재사용,유효시lite23전체검증.

- seed2 scalejitter fold1 완료: 학습/추론exit0·입력SHA불변·추론메타동일·18records로컬재합산일치. HR78.014472124%,대응CE대조대비-.107071593%p;고정E78.081397519%,CE대조대비-.077000778%p. 둘다악화로확대기각. 결과/로그/receipt수집,공식validtest미사용/lite23기준유지/81%미달성.

### Codex seed2 photometric off fold1 — 실행 전 등록
- ID `accuracy81_HR_seed2_photometricoff_fold1_20261010`, GPU0 실제260MiB/0%·공유잠금없음 확인 후안전controller실행. 원본seed2fold1 동일4epoch/seed819349/CE Dice경계/BN고정/기하증강유지. 밝기대비·감마·노이즈·블러p만0,transform node보존으로기존pernode RNG인덱스유지. 밝기/질감증강제거효과held18 독립HR·고정E 대조,공식validtest미사용/Claude중복없음/원본보존. 유효시lite23전체검증.
- round3 완료:578개22이하 축소후보 전부품질gate미통과. original27/lite25/raw23/canonical23 대조exact/전체점수gate순수Python 독립재집계일치. 최고평균21개s1_4_s2_4 80.546629914%이나최악−0.375340472%p(vs23)로기각. 결과/완료/score로그 로컬수집,기존lite23보존.

### Codex lite23 E 가중치 FP16 저장 — 실행 전 등록
- 사용자추가경량화 후속. ID fusion805_storage_fp16_20261010. 23구성유지/E B4 4+HR 4의float32 tensor를FP16저장,BN running mean/variance 및int buffer 보존. 추론시기존FP32망에로드/FP32계산강제. R14/F/규칙동일. 성능저하가능성 있으므로모든원본/sha보존·별도assets생성,변환파일roundtrip tensor exact검사. 저장용량경량화이며추론속도나FP32실행메모리감소는주장하지않음.
- CPU생성후GPU0유휴·공유lock확인하여원본E20재추론→기존검증rawR/F결합→독립PNG채점. mean≤.01/worst≤.1%p original27/lite25/lite23 모두gate. 기존GPU0타작업간섭없음/현재다른작업사용중. 원본lite23ZIP 및기준SHA불변.
- FP16저장후보CPU생성완료:23checkpoint 총3,317,816,946→2,540,165,234bytes, E8개변환/BN통계·원본파일SHA보존/변환후tensor roundtrip exact. 추론FP32강제assert 포함. build완료/manifest로컬수집. 성능은아직미검증. controller2432303 실제ps생존, GPU0 기존타작업4308MiB/95%이라15초간격유휴대기→공유lock→E원본20→F/독립채점 자동진행. 기존lite23패키지불변. 이번goalturn은round3종료증거수집/전체gate독립재집계/다음저장경량화실물생성/검증queue생성의progress.
- FP16후속goalturn: controller2432303 실제ps생존확인, GPU0 기존타작업4308MiB/95%로대기유지. 이전turn은변환실물생성/검증queue로progress. 이번turn은저장경량화 재생성도구를portable CLI(--parent-package/--original-assets/--refiner/--output)로준비, 실제서버생성소스 별도보존. 기존infer_compact metadata복원/입력·가중치SHA검사wrapper를새후보에연결. 실행중controller/동결될추론소스불변.

- photometricoff GPU0 외부작업진입으로안전감시기가우리child만SIGTERM,controller/child/score 실제종료확인. 외부NVML3505266 7382MiB사용/기존2123683보존. epoch1 550이후저장전중단,original에는args.json만있어resume불가. 실패기록/로그수집보존. 새ID `accuracy81_HR_seed2_photometricoff_fold1_20261010_retry1`,동일고정레시피별도출력·GPU0유휴waiter등록. 대기중재학습없음,외부작업종료/우회없음.

### Codex lite23 HR confidence cascade — 점수생성 전 등록
- GPU0FP16원본검증은외부작업점유로대기/controller2432303 실제생존. 사용자우선순위옵션답변없으므로용량·계산량둘다줄이는방향진행.
- 별도CPU 진단 ID fusion805_confidence_cascade_20261010. 첫E=B4 4+HRseed1fold4+seed2fold4(21호출), 불확실시seed1fold2/seed2fold1 추가해원래23E 복구. F1회/R14고정. 23가중치보유/HRforward평균감소가능성확인하며속도실측아님. FP16새가중치와아직결합하지않음.
- 입력conf confidence규칙12개고정:HR양시드라벨불일치율t=.01/.03/.05/.10,확률max<.8인픽셀율t=.01/.05/.10/.20,평균정규화entropy t=.10/.15/.20/.25; feature>t이면full23. GT없는decisions생성후이전독립PNG per-image IoU로별도gate계산. ≥5영상cheap 및 mean≤.01/worst≤.1%p(original27/lite25/raw23/cache23) 모두통과만후속. 재탐색편향/새forward미검증명시. 기존패키지/가중치불변.
- cascade CPU사전비교완료:entropy>.10에서full복귀하는규칙이5/20영상cheap/평균HRforward3.5(4대비−12.5%)/IoU80.5187854924%,최악vs23−.018567831%p로모든gate통과. ≤4cheap규칙도통과하나사전≥5cheap조건에따라entropy.10만후속. 23가중치보유/새forward및시간미검증/반복valid선택한계.
- 별도cascade GPU0실제실행코드준비:cheap의FP32 raw HRfold4를유지하여full복귀시FP322fold평균뒤FP16가족반올림(원래23수식보존). 고정FP16 저장검증 통과후에만같은FP16assets로GPU0원본20cascade→F→PNG채점. full FP16과동일GPU20영상 E 시간/실제HRforward기록비교, R/F시간제외한1회짝비교임을명시. 대기중fixedFP16 infer에는시간·HRforward4계측만추가(예측수식불변);source snapshot생성전임확인. 기존lite23/원본불변.
- cascade 조건부시편제외선택 진단:선택entropy.1/전체재선택held mean80.5000468786%,최종23대비−.017223593%p(평균기준미통과),HRforward평균3.5. 독립일반화미검증/고정valid사전비교와구분. 실제cascade코드구문/CLI확인완료; controller2434215 실제ps생존,고정FP16 controller2432303 완료·품질통과 후에만GPU0공유lock/유휴→20영상→독립채점/동일GPU시간계측 진행. 기존모델채택변경없음.
- GPU0후속검증 차단감사: FP16생성turn/portable준비turn/cascade준비turn 연속3goalturn 동일외부GPU0점유 확인. fixed controller2432303/cascade controller2434215 실제ps생존 및오류없음, GPU0약9.3GB외부점유. 저장변환/조건부실행코드/사전비교/후속자동채점 준비완료; 남은최종raw검증은GPU0유휴라는외부상태변화필요. goal은완료아님/차단상태기록,authorized 대기·후속검증queue는유지. 기존lite23패키지/원본그대로.

- Claude `claude_prune_seed456_m109_20261011` (CPU 전용, 서버 `/workspace/model349_compress/claude_prune_seed456_m109_20261011/`): lite23_b4ws에서 R 고해상도 밝기 구간 중 seed456(152.3~152.7)·model109(174.6~175.1)를 restart51로 보내 두 가중치 제거(23→21). 영향 이미지는 valid A40·A74·A11_step2·A67뿐, test10은 해당 구간 이미지 없음(채널0 평균으로 확인) → test 마스크 불변. R 라벨은 exh14 저장 확률(slim)로 오프라인 생성(기존 route 재현 시 release 라벨과 1~6px 차이), 변경 픽셀만 release 라벨에 반영 후 기존 E(b4ws)·F(epoch8)로 infer_fusion 재실행. control 80.54643157686947 정확 재현, pruned **80.52876956%** (−0.01766%p; A74 −0.19, A11_step2 −0.13, A40 −0.03, A67 −0.00). R 단독 −0.0192%p(저장 탐색). 판정 보류(사용자 결정). 학습·GPU 사용 없음.
- Claude lite21 배포(2026-10-11): 위 제거안을 `Model349_lite21_b4ws_bundle_code.zip`(CNUKoMaP/lhs fdeeccf, SHA256 9b00beac…)로 업로드. `infer_lite.py`에 선택 설정 `inference.pruned_high_bands` 추가(미설정 시 lite23 동일), `run_infer_lite21.sh`/`selected_*_lite21.*`. 수정 코드로 영향 4장 GPU7 원본 추론: 4장 모두 restart51, 오프라인 R 라벨과 1~5px. valid20/test10 전체 e2e 재실행은 안 함. lite23 ZIP 불변.
- Claude 추가 제거 스캔(lite21 기준, R 라벨 오프라인+E(b4ws)·F 결합, valid20 최종): 322f4 제거 **+0.0016**(최악 −0.008), 322f3 −0.004, 322 전체 −0.009, pairwise 구간 −0.010(test 2장 영향), 347 −0.025, 326 f1/f2 −0.042/−0.046, 329 f1/f3/f4 −0.28/−0.07/−0.08, 269 −0.095, 336 −0.385. 조합: 322f4+pairwise −0.008, 322+pairwise −0.021, 322f4+347 −0.023, 322+347 −0.028. 결과 `results_lite21/`. 반복 valid 선택 편향 동일.
- Claude lite20 배포(2026-10-11): `Model349_lite20_b4ws_bundle_code.zip` (CNUKoMaP/lhs f47ade8, SHA256 60cd2b9d…). lite23_b4ws − R model109·seed456·322fold4 = 20개. GPU7 원본 R 추론(valid20/test10) + 기존 E(b4ws)·F 결합: valid20 **80.5301323%**(오프라인 예상 80.5304, R 라벨 44px 차이), test10 lite23 대비 836/9,074,964px. lite21 test R은 lite23 release와 11px, 최종 2px. lite19 R valid 80.1922(오프라인 일치, 최종 결합 미실행). infer_lite: GPU 점유 검사 시작 시점만(공유 시 경고 후 계속), run_infer.sh GPU 번호 버그 수정(이전엔 물리 GPU0 사용). 보고서 점수는 시드 평균으로 별도 기재 예정(사용자).
- Claude 밝기 구간 규칙 단순화 진단(2026-10-11, CPU, lite20 기준, 저장 확률 R + E(b4ws)·F 결합): route_params의 좁은 구간 보정(beta 15: 164.0~164.5·174.6~174.85, eut_bias −2.0: 161.3~162.0, −0.75: 164.5~165.2, +0.3: 174.0~175.5)을 기본값(beta 3/7 @160, eut −1.0/−0.3 @165)으로 바꾸면 valid20 최종 −0.153%p(A11_step1 −1.09, A186 −1.04, A153 −0.62, A11_step2 −0.18, A67 +0.33). pairwise 구간(159~161, restart61+전용 파라미터)까지 없애면 −0.470%p(A71 −6.34). test 영향: A87(164.74)이 eut −0.75 구간, A167·A51이 pairwise 구간. 판정: 단순화 미채택, 보고서에 "밝기 구간별 보정값은 검증 데이터로 선택" 명시(사용자 방침). 서버 `claude_prune_seed456_m109_20261011/scan4_result.json`.

## Codex ANA train70 B4 SWA — 실행 전 등록 (2026-10-12 17:12 KST)

- 사용자 승인: 현재 E OOF 추론 → 선택 → valid 채점 뒤, HRNet392 시드 학습 전에 추가.
- ID: `codex_r70_swa_20261012`; 서버 `anacnu:~/komap/codex_r70_swa_20261012`; ANA RTX5080 GPU0만 사용(사용자가 이번 큐 삽입 승인).
- 기존 train70 seed0/21,000 step의 322·326·329 가중치에서 순차 SWA 추가 학습. train70만, patch1536/batch2, 원래 입력·정규화·샘플링 유지. 4,000 step, 고정 LR1e-4, 매500 step 평균(8개), BN train300 batch 재계산. valid로 SWA 설정 선택하지 않음.
- 기존 `r70/train_swa.py`를 복사해 사용. 원본 가중치/릴리스와 Claude OOF 작업 보존. 기존 체인의 `run_e.sh` item3 직전에 일회성 잠금/완료 확인 게이트 삽입 예정. 진행 중인 E OOF 작업은 중지하지 않음.
- 평가: lite18h의 R B4만 train70 또는 SWA 가중치로 교체하고 저장된 나머지 R/E 확률·F 재사용, 최종 valid20 짝 비교. 결합 비율 고정, test GT 주장 없음. 결과에 기존 fold 대비 및 train70 대비를 별도 보고. 자동 릴리스 교체 없음.
- 상태: **큐 연결 완료** (2026-10-12 17:13 KST). 실제 실행 중인 `chain_after5.sh` PID2513950은 E OOF probe 단계. 그 체인이 호출할 `eseed/run_e.sh`의 item3 바로 앞에 SWA 게이트 삽입·재확인. 기존 스크립트 백업, flock 잠금 및 완료 파일로 중복 방지. Python 구문/CLI import와 bash 구문 확인 통과. SWA GPU 학습은 아직 미시작. SWA 실패 시 failure.json 기록 후 원래 HR 시드 체인은 계속 진행한다.
- 동시 확인: lite18h fresh valid20 **80.54622071667629%**. offline과 valid20 최종 라벨 0/15,757,260px, test10 0/9,074,964px 차이. `l18h/e2e.out`의 L18H_E2E_DONE 확인.

## Codex ANA restart61에 model269 학습 방식 적용 — 실행 전 등록 (2026-10-12 17:24 KST)

- 사용자 승인: 현재 큐 4번 E OOF 설정 valid 평가 뒤, B4 SWA 전에 실행.
- ID `codex_restart61_globalstats_20261012`, ANA `~/komap/codex_restart61_globalstats_20261012`, RTX5080 GPU0. 학습/추론 중인 E OOF와 중복 실행하지 않음.
- 부모는 공개 restart61, 원래 ASPP U-Net++ 구조 유지(269 가중치/보조 경로/증류 미사용). train70 정답만, 원본 배율 patch512와 2배 patch768를 1:1 사용. raw/Z/CDF는 전체 resized 이미지에서 계산 후 crop, 학습·추론·TENT 입력 모두 일치. 10epoch, 이미지당8개, seed819269, batch1/accum2, AdamW LR3e-5(encoder0.2), wd1e-4, BN 통계 고정, 원래 CE/Dice+pairwise loss 유지, 매 epoch 가중치 저장, 마지막 10epoch 가중치만 최종 평가. valid로 학습/epoch/설정 선택 없음.
- 기존 lite18h 고정 R/E/F로 비교: 기존18h, 기존18h에서269만제거, 추가학습restart61로교체+269제거(17개). 필요 보조 비교로 추가학습restart61+269유지. E OOF 선택 후보는 혼합하지 않고 기존 공개 E/F 고정. valid20은 개발 점수이며 test 정확도 주장 없음. 자동 릴리스 교체 없음.
- 큐: E OOF 평가 → 본 실험 → B4 SWA → 기존 HRNet392 시드 체인. 원본/Claude 기록 보존.
- 상태: **큐 삽입 완료** (2026-10-12 17:28 KST). `eseed/run_e.sh` item1 앞에 일회성 flock 게이트 삽입, SWA 게이트는 item3 앞 유지. chain_after5.sh PID2513950은 아직 E OOF 추론 중(hr_s2_f3). 원본 run_e.sh 백업 보존. CPU strict parent load, train70 쌍/입력 shape, 모델 forward/loss/backward 유한값 및 bash 구문 검증 통과. 실제 GPU smoke는 기존 작업과 겹치지 않도록 큐 진입 후 실행 단계에서 확인. 실패 시 failure.json 기록, 후속 SWA/HR 체인은 계속. 후보 실패/미완료를 성공으로 처리하지 않음.

- 사용자 변경 반영 (2026-10-12 17:31 KST): **10epoch + 매 epoch 체크포인트**. 학습 시작 전 실제 PID/출력 상태 확인 후 수정했으며 4epoch 코드 백업 보존. 총5,600조각/2,800 optimizer update, 스케줄 LR는 기존 고정값 유지. `checkpoints/epoch01.pt`~`epoch10.pt`와 epoch별 train loss/시간/SHA 기록. 중간 valid 채점·best epoch 선택 안 함. 체크포인트 가중치 약1.13GB 추가 저장 예상. 서버 AST 및 변경 코드 확인 통과.

- 사용자 epoch 선택 변경 반영 (2026-10-12 17:36 KST, 위 마지막epoch만평가 방침 대체): 1~10epoch을 고정된 no269 R→E→F valid20으로 순차 채점, 이전 최고를 엄격히 마지막으로 갱신한 epoch 선택. 모든 epoch와 10epoch 점수 함께 기록. valid로 선택한 개발 결과이며 독립 일반화 점수 아님. 기존18h보다 좋지 않으면 개선/채택으로 주장하지 않음.
- 등록 코드 변경 직전에 trainer PID2833212가 17:33:13 시작했으므로 이미 메모리에 로딩한 10epoch 학습은 중단/재시작하지 않음. 원래 마지막epoch 평가 뒤, SWA 실행기 진입 시 `--select-epochs` 후속을 먼저 실행하도록 연결 확인. training_complete로 재학습 방지, epoch10 기존 확률 캐시 재사용, 기존 마지막epoch 결과 별도 백업. 선택 완료 파일은 epoch_selection_completion.json. Python AST 및 후속 연결 확인 통과.
- 동시 실제 확인: E OOF 12개 추론 EOOF_PROBE_DONE, CPU 선택 EOOF_SELECT_DONE. 0.5 B4 비중 유지+덮어쓰기 class(0,2,3) 선택: OOF stage1 79.9709→80.1800(+0.2091), 바깥fold 추정80.0790. valid20 최종80.5462→79.7010(−0.8452), 최악−13.663%p, 개선13/악화7. 본 restart61 실험과 SWA는 공개 E/F 설정을 유지함.

## Codex 팀원 seed100 자료 감사 및 lite18h E 축약 CPU 확인 — 실행 전 등록 (2026-10-12 19:38 KST)

- 사용자 요청: CNUKoMaP/giveItToHyeonSeok 시드 편차와 축약 가능성 확인. 읽은 커밋 b59fc743d6aca93acf1234a9ce3819076afa7551. 100회 계획 중 제공된 결과는5계열28후보/56가중치, 모델 LFS 실가중치는 이번 감사에서 다운로드/재추론하지 않음.
- 팀원 기준은 재학습 lite20 80.2558455%, 현재 공개 lite18h 80.5462207%와 직접 비교/효과 합산 안 함. HR347은 부모별4개씩 나눠 분석. 28후보 클래스→이미지→평균 산술 일치 확인.
- ID codex_team_seed_review_20261012. CPU만, 서버 ~/komap/codex_team_seed_review_20261012. 실행 중인 Claude HR seed GPU 학습 유지.
- ANA 기존 eoof/e_prune.py에 정의된 E B4 2/4 6조합, HR 시드당1개 4조합, 동시24조합 + baseline을 저장 확률로 재검산하고 결과 보존. 해당 스크립트 실행 중인 PID 및 저장 결과 파일은 현재 확인되지 않았음. 원본 스크립트/캐시/가중치 보존, 별도 복사본 사용. 반복valid 선택 편향 명시, 자동채택/릴리스 교체 안 함.
- 상태: **감사·CPU 비교 완료** (2026-10-12 20:03 KST). 28후보 원시 class IoU 산술 재계산, source_snapshot.zip SHA 일치. 시드 표본SD(pp): R322 n4 0.002156, R326 n4 0.022653, E B4ws n6 0.002779, E HR392 n2 0.003372. HR347 부모별n4 SD 0.000717/0.001890/0.000344로 부모 변경 효과와 분리. R329 결과는 저장소에 없음.
- 현재 lite18h 저장확률 CPU35후보 대조군 80.54622071667629% 재현. E B4 f1·4만 유지(전체16개):80.56021571869982%, +0.013995, 최악−0.093186. E HR s1f4·s2f4만 유지(전체16개):80.59099445442047%, +0.044774, 최악−0.208540. 둘다(전체14개):80.59103872576644%, +0.044818, 최악−0.357404. 전체 조합 평균 변화: B4−0.019381, HR−0.037031, 동시−0.051172. 최상만보고 일반화주장안함.
- 선택후 조건부 시편bootstrap95%: B4[-0.015699,+0.049068], HR[-0.003175,+0.093509], 동시[-0.025815,+0.115531]%p. 모두0포함하며 반복valid 선택을 교정한 구간 아님. fresh/test 새추론 미실행, 공개가중치 자동교체 안 함. 로컬 결과 seed_statistics.json/csv, prune_scores.csv, prune_selected_bootstrap.json 및 팀 메타데이터 보존.

## Codex 국소 손실 우선 E B4 2fold 실험 — 실행 전 등록 (2026-10-12 20:15 KST)

- 사용자 승인: 국소 손실을 작게 유지하는 방향의 축약 실험. ID codex_lite16_b4_local_20261012, ANA ~/komap/codex_lite16_b4_local_20261012.
- 후보 E B4ws fold1·3 및 fold1·4, HR392 4개/F/R은 lite18h 그대로. 각 후보16가중치. 평균≥기준과 최악 이미지 mIoU 손실≤0.10%p를 screening 기준으로 기록, 클래스별 손실/픽셀 변화도 함께 보고. valid 선택 편향 명시. 자동 릴리스 교체 없음.
- 현재 ANA RTX5080은 HR392 seed6 s2f1 학습(PID3651925)중. 기존 chain_after5.sh PID2513950, run_e.sh PID2833208 및 GPU compute 작업 종료 후에만 fresh E 추론(대조군4fold와 두후보2fold)실행. 원본 R 출력은 이미fresh검증됐으므로 SHA·메타데이터 확인 후 재사용, 중복 R 추론 안 함. 공통 HR4개 추론은 세 구성에서 공유, F는 각 구성별 CPU 재계산. valid20/test10 원본이미지 추론, test 점수 주장 없음.
- 기다리는 동안 저장 확률로 대조군/두후보의 클래스별 IoU·worst 이미지/픽셀 변화 계산. 기존 E pruning 기록과 중복 GPU 작업 없음. 원본 캐시/가중치/릴리스/Claude chain 보존.
- 상태: **CPU 상세 검증 완료 / fresh GPU 큐 대기** (2026-10-12 20:22 KST). waiter PID3815131, flock PID3815130, queue_status.json 확인. 학습체인/기존 compute가 끝난 뒤20초 간격3회 idle 확인 후에만 실행. valid/test 공통HR 추론은 대조군/두후보에서 공유, 기존 R은 중복추론 안 함. 원본 전체 R/E/F 체크포인트·메타데이터 SHA와 valid/test R config 일치 검증 통과.
- CPU 대조군 마스크차이0/15,757,260px. f1·3:80.554509%, +0.008288%p, 최악이미지−0.082220, 최악이미지클래스−0.341839, 변화36,861px. f1·4:80.560216%, +0.013995, 최악이미지−0.093186, 최악이미지클래스−0.640628, 변화42,086px. 국소손실 우선 후보는 f1·3이며 f1·4 대조도 유지. 새 GPU 추론 결과는 아직 없음. 자동 릴리스 교체 없음.

## Codex ANA R326 시드 편차 안정화 — 실행 전 등록 (2026-10-12 20:52 KST)

- 사용자 승인: cnuana/anacnu에서 진행. ID codex_r326_ema_seed4_20261012, 서버 ~/komap/codex_r326_ema_seed4_20261012, RTX5080 GPU0.
- 순서: 기존 HR392 seed 체인 → codex_lite16_b4_local fresh 검증 → 본 실험. 실행 중 작업 중지/중복 실행 안 함.
- 같은 model303 specimen fold1·2 사용, 기존 prepare_oof_splits 생성 trees의 train/heldout 그룹을 model303_progress와 대조. 시드세트4개: (3455,3456),(3457,3458),(3459,3460),(3461,3462). 기존 팀원 seed 탐색에 대응하지만 기준은 고정된 공개 lite18h(R9/E8/F1). R326만 교체, E/R나머지/F/결합비율 고정.
- 원본326 레시피: B4 U-Net, zcdf3/image_global, scale3/patch1536/area/batch2,16,000step,AdamW lr3e-4/wd1e-2,CE+Dice,300stepwarmup+cosine,workers4,valid 학습중평가없음. 학습후반12,000step부터 EMA .999, 마지막4,000update 평균. raw/EMA를 동일 run에서 보존. BNbuffers는 raw에서 복사, 별도 train-only300batch BN재계산 plain_bn/ema_bn 대조도 저장. BN재계산 입력 RNG는 fold별고정·시드간동일,dropout/drop-path eval. 추가배포가중치없음(326은fold1·2 두개).
- 평가: 4시드세트의 plain/EMA/plain_bn/EMA_bn 최종lite18h R→E→F valid20 평균/표본SD/범위/최악이미지·클래스/짝차이, fold1·2 단독교체 대조로 변동 기여도 확인. 평균을 낮춰 SD만 줄이면 성공으로 주장하지 않음. valid 시드best선택/학습중loss로개선판정/자동릴리스교체없음.
- 예상: 학습8개 순차 약10~13시간 + 평가,기존GPU큐대기는별도. 상태: **큐 연결 완료**. flock PID4026701, 기다리는 실행기 PID는 queue_status.json에 기록. 원본 model303 fold1/2 그룹·이미지수·이미지/마스크 SHA 검증, CLI import/AST, EMA CPU 파라미터 평균·BN버퍼복사 검증 통과. 현재 학습 미시작이며 기존 HR 및 lite16 검증 큐를 앞서 실행하지 않음. CPU 평가 workers는 spawn 사용, GPU idle 검사에서 자기 실행기 PID만 제외해 context 잔류로 자기대기하지 않도록 처리.

### Codex GPU 실제 확인 — 2026-10-12 22:16 KST

- lite16 B4 후보 fresh 검증 **완료**, 원본R 재사용 SHA/config 확인 및 fresh E/F valid/test 수행. 대조군 valid80.54622071667629, valid/test 마스크 변화0으로재현. B4f1·3:80.55464328764197%, +0.00842257%p, 최악이미지−0.08256555, 최악이미지클래스−0.34145931, test 변화20,299/9,074,964px. B4f1·4:80.56007857046181%, +0.01385785, 최악이미지−0.09335589, 최악이미지클래스−0.64100545, test 변화21,918px. 둘다평균·최악이미지screening통과. 국소손실우선 f1·3 후보 유지, 자동릴리스교체없음. 로컬fresh_result/fresh_completion수집.
- R326 EMA 학습 **실행중**: controller4026702, training child129891, set0 fold1 seed3455,5,500/16,000step(첫학습34.4%,전체128,000step4.3%).0/8개완료. EMA는12,000step부터시작예정. RTX5080 GPU99%,메모리14,659MiB.관측속도약0.29초/step,남은학습약10시간+평가. 중간loss로성능판정안함.
- Claude HR392 후속채점 결과파일존재하나 hr5 seed2fold4는31/70epoch중단체크포인트이므로그세트는완전한재학습시드점수로취급안함.
- P100 현재SSHnamespace에서우리KoMaP학습·추론프로세스미확인.GPU3는13,248MiB점유/관측util0%,GPU6/7사용중.소유자추측/종료/재실행없음.

## Codex lite16 기준 채택 — 사용자 승인 (2026-10-12 22:30 KST)

- 사용자 지시: fold 축약 lite16 체제로 전환. 채택 구성은 국소손실이작은 **E B4ws fold1·3**, 기존 R9/HR3924/F1 유지, R9+E6+F1=16. 이름 lite16_b413. fresh valid20 80.55464328764197%, 기준18h대비+0.00842257%p, worstimage−0.08256555%p, test 변화20,299px(정확도아님).
- 원본lite18h 가중치·코드·예측 보존. 별도 ANA ~/komap/lite16_b413_adopted_20261012 에 채택 config/16개manifest/2fold지원추론코드와 adoption receipt 작성. live학습/공유코드원본수정없음.
- R326 EMA 학습은 재시작하지 않고 원레시피 유지. 사전등록 lite18h 평가도 유지하며, 완료 후 저장확률로 lite16 고정E/F 기준 CPU 짝평가를 별도 추가해 두기준결과구분. 추가 GPU학습없음.
- 실제학습22:28확인:first set0/fold1 seed3455 7,900/16,000step(49.4%,전체6.2%),RTX5080 util99%,14,659MiB.8개중0완료.12,000step부터EMA시작.학습loss로성능판정안함.

- lite16 기준 채택 구현 완료: ANA/로컬 config·16checkpointmanifest·2fold지원 infer_expert/infer_compact·run_lite16.sh·ADOPTED_BASELINE.json 보존.16개총2,424,376,128bytes. CLI/AST/bash구문및count/fold확인통과. 기존18h코드/가중치불변.
- 진행중326사전등록18h평가는수정하지않고,후속 CPU lite16 평가큐 별도로연결(PID351745,flock351744).완료된326개별확률재사용,원학습완료뒤실행,새GPU학습없음. 결과lite16_result.json에별도기준저장.

## Codex HR392 세트5 미완료 fold 재개 — 실행 전 등록 (2026-10-12 22:45 KST)
- 사용자 지시의 hr329 세트5는 직전 대화의 HR392 세트5로 해석. ID codex_hr392_set5_resume_20261012. ANA GPU0, R326 EMA 전체 학습·추론·원평가 및 lite16 후속평가 완료 뒤 실행.
- hr5 seed2 fold4의 train_state.pt epoch31을 별도 디렉터리로 복사해 optimizer/scheduler/scaler/RNG 포함 resume, 동일 seed5002/70epoch 레시피로32~70epoch만 실행. 기존 hr5/다른7개 가중치·캐시 보존, 원본 덮어쓰기/재학습 없음.
- 완료 후 세트5 HR4개를 lite16_b413 고정 R/E B4/F에 결합하여 valid20 fresh E/F 평가. 검증된 R release 재사용. 새 결과 별도 보존, 반복valid개발점수/자동릴리스교체없음. GPU 점유 시 대기, 타작업 종료 없음.
- 큐 실행기 연결 완료. 별도 flock 잠금, EMA 전체완료 및 lite16평가 완료를 모두 기다리고 GPU 유휴3회 확인 뒤 시작. train_state의 model/optimizer/scheduler/scaler/RNG 저장과 epoch31 확인. 원본출력 대신새training 디렉터리 사용. 수정 검증 중 대기 실행기만 교체했으며 실제 학습/다른 사용자 작업은 중단하지 않음.

## Codex lite16 E 재축약 CPU 실험 — 실행 전 등록 (2026-10-12 23:02 KST)
- 사용자 승인 HR392/B4ws 축약 재확인. R329 시드탐색은 팀원 담당으로 중복하지 않음. ID codex_lite16_E_rescan_20261012, ANA CPU만. 기존326 GPU/HR392세트5 대기큐 보존.
- 현재lite16 fresh E B4ws fold1·3을정확대조군으로사용. B4 (1,3)/(1)/(3) × HRseed1 (2,4)/(2)/(4) × HRseed2 (1,4)/(1)/(4) =27개구성, 현재16개에서15/14/13개 후보. 가족비율.5/.25/.25 및R/F고정. 저장된fold확률차이를fresh E에반영하는CPU screening이며반올림으로fresh후보와미세차이가능,control마스크exact필수.
- 모든후보평균/최악이미지/최악이미지클래스/클래스평균/변화픽셀보고. 기존mean≥base 및worstimage≥−.10%p 기준 유지. 선택valid편향명시,새학습/추론/자동릴리스변경없음.
- CPU27개완료/로컬summary·per_image·completion수집. control80.55464328764197% 및20장마스크exact. 대조군외26개 모두기존mean≥base/worstimage≥−.10%p gate미통과. B4fold1only80.53590684(−.018736,worst−.318143),fold3only80.48675432(−.067889,worst−.555652). HRseed2fold1제거(15개)80.56767284(+.013030,worst−.101660,이미지클래스worst−.261276)로경계근처이나기준미통과/FP16캐시screening이며fresh확정아님. HRseed1fold2제거(15개)80.58332231(+.028679,worst−.175165). HR양시드fold4only(14개)80.58783533(+.033192,worst−.274887). 13개최고B4f1+HRf4/f4 80.56288859(+.008245,worst−.612976,이미지클래스worst−2.269851). 평균상승과국소손실구분/lite16채택유지/다른작업변경없음.

## Codex test10 lite20→lite16 변경 영역 시각 감사 — 실행 전 등록
- 사용자승인: 버전별test10변경영역직접비교. ID codex_test_lite20_to16_20261012. 서버기존최종예측/원본/manifest수집 후로컬CPU 픽셀전이·경계/연결성·원본대조crop시각검토. GPU학습/추론재실행없음,326/HRqueue보존. test GT없으므로정확도/개선확정주장안함. 존재하는버전만비교하며lite17결과임의생성안함.
- 시각감사완료: lite20/19a/18/18h/16 test10최종라벨·원본수집,20↔18h입력SHA일치.10장각최대4고밀도변경crop대조,HTML갤러리/report/comparison보존.20→16변경44,633px/.4918%,18h→16 20,299px/.2237%.기존경계주변(5×5팽창)99.49%/99.55%.20→18h에서A122224px/A51161px/A87194px작은PrimarySi영역이주로Eutectic으로재분류된것발견,16에서도유지/B4ws축약기원아님.18h→16완전소실10px이상4연결기존영역7개최대22px,Al3Ni/Eutectic;큰입자전반형태유지/미세경계·연결변화있음.원본그레이만으로상의정답불확실/test정확도개선주장안함. lite18은별도가지만비교포함/lite17결과미확인.

## Codex lite16 R→E→F 단계 오류 진단 — 실행 전 등록 (2026-10-12 23:20 KST)
- 사용자 승인 1번 단계오류분석 진행. ID codex_lite16_REF_errors_20261012. ANA CPU 및 기존fresh R/2foldE/F valid20예측만사용,학습/추론반복없음. GPU326/HR세트5큐보존.
- R/E stage1/F final별per-image4IoU 및 단계간 수정/신규오류/틀림→다른틀림 픽셀,GT클래스별분해,GT경계2px내외분해. 최종fresh80.55464328764197재현필수. 각이미지E/F신규오류밀도최대영역원본/GT/단계예측crop생성·시각검토. valid반복개발진단,새설정선택/릴리스교체없음.
- 단계진단완료/로컬summary·per_image·40crop·HTML·report수집. R79.79930890→E80.50419733(+.70488843)→F80.55464329(+.05044596)%. E20/20이미지개선,F15개선5악화. E수정166329/신규105389/오답변경24426,F수정12757/신규7278/오답변경661. F신규오류GT Al3Ni3181/Primary390/Eut3707;99.75%GT경계주변. 클래스IoU모두순개선(침범과거짓양성감소동시). F최악A170−.0390713pp. E/F통째로제거근거없음,새모델보다Al/소수상경계충돌분리정보진단후설계권고. GPU큐불변/새gate설정선택없음.

## Codex lite16 보완성·R/E혼합·E/F규칙 CPU 연구 — 실행 전 등록
- 사용자승인 후속2/3및규칙/혼합/오류완화. ID codex_lite16_complement_rules_20261012. ANA CPU만/기존326학습HRqueue보존/R329팀작업중복없음.
- HR4fold각argmax오류의다른fold보완및4개제거짝분해(A153확대),고정16기준캐시. 기각N1 HR3배/N2 ConvNeXtTiny/N3 HRzcdf/N4 B4scale4/EutL2 기존v1확률만검사(없거나class/입력검증불가시제외).
- 규칙사전범위:E margin≥.05/.10/.20,E max≥.6/.7/.8,HR4+B4family정답없는투표≥2/3, F margin≥.02/.05/.10,F Al확률증분≥.02/.05/.10,F expertAl≥.4/.6, F family투표≥2/3; E/F게이트결합세개. 규칙은원본확률에서만계산/GT읽기후변경없음.
- R확률은저장구성에서원래보정logit복원후softmax(후처리freshR라벨과차이기록·라벨정렬),R/E .1/.25/.5 균일확률혼합 및충돌픽셀혼합. 혼합E를F에입력한경우와F기존출력유지경우분리. R/E확률미교정한계명시.
- 20개perimage/class/국소gate/조건부시편제외선택(기준20장중동일시편2장제외그룹19개)진단,全候補보고. 반복valid선택편향교정못함/자동채택없음/test추가선택안함.
- 분석중326後평가가참조하는freshR/probabilities경로누락발견. 이번20장R복원과freshR라벨차이0px검증한l18h/R_valid/probabilities를freshR/probabilities에읽기용symlink연결(원본예측/기존파일불변/학습실행기코드수정없음). 누락경로로인한후속실패방지. 원보정logit→후처리라벨차이7810px는혼합연구에서만fresh라벨정렬후정규화했으며교정된확률로주장안함.
- 연구완료: 최초52+사후규칙조합3+HRfold교체8=63고유구성. 대조군20장마스크exact80.55464329. N3(HR336→zcdf25ep,seed414) E10%추가80.60349850(+.048855,worst−.072903)·L2 E10%80.59832946(+.043686,worst−.063083). R347교체는모두악화. N1Eut.25 +.020014/worst−.069093,N2/N4기각.
- 16개수유지HRs1f2→N3교체80.62272620(+.068083),17개선3악화/worstA72−.084313,최악이미지클래스−.271310. 복구27763/신규18926px. 4클래스평균모두개선. 조건부시편CI[+.031222,+.109319],19시편제외선택19/19동일후보(독립일반화아님). L2→HRs2f1도80.60101979(+.046377,worst−.075833). cache차이보정/가족FP16rounding로fresh값미세변경가능/새후보test未検証/現lite16유지. recipe JSON 및전체결과/crop/HTML/report보존.
- HRs2f1 단독다른3fold오답시유일정답50,494px(최종기여와구분). A153제거신규오류2273/복구1048,손실GTAl1949/Al3Ni304/Primary8/Eut12;그중1793px에서s2f1도GT일치. s2f1중복간주부적절.
- E≥2/5동의+.007488pp/worst−.089054, F E-Al≥.4 +.002503/worst−.032062(총오류픽셀52증가). 사후두규칙결합+.010260/worst−.060516. R10%E90%+F재계산+.002980이나조건부시편제외선택−.008419pp;직접혼합권장안함. 소규칙CI0포함. 次優先検証N3s1f2교체fresh valid/test 및재학습시드대조,자동채택/GPU추론예약없음. 81미달.

## Codex N3/L2 단독·동시 교체 test10 비교 — 실행 전 등록
- 사용자승인N3/L2각각및동시교체test변화측정. ID codex_lite16_N3_L2_test_20261012. 고정lite16 R/E B4/F 유지,N3→HRseed1fold2,L2→HRseed2fold1,양교체모두16개. test GT사용없음/변화량및시각비교만.
- ANA326EMA와기존HR392세트5재개완료뒤GPU0idle실행. 현재ANA99%14659MiB,P100허용GPU0/3각12994/13248MiB점유로타작업간섭안함. N3/L2test캐시없음확인.
- N3legacyweightmodel414+기존ZCDF PNG test입력,L2model_eut_L2+raw RGB 입력복사·SHA확인. 기존R/E/F/raw SHA보존. 부모HR3924개fold raw FP32는가족평균FP16교체식정확복원을위해필요한캐시만생성,새학습없음. 같은model/이미지한번추론결과를3후보가공유. 원래B42/R재추론없음. F3후보별CPU재계산.
- 완료시baseline마스크exact검사/변경픽셀·class전이·연결영역소실·고밀도영역원본대조PNG/HTML및report자동생성. valid점수상승을test정확도로주장안함.
- N3/L2 test큐연결완료: waiter796571/flock잠금,앞선326전체평가및HRseed5완료후실행. N3/L2가중치+ZCDF test10 총12파일원본↔ANA SHA exact확인;두모델strictCPUload/64pxforward정상,N3dataset10개[3,H,W]확인. 아직test추론시작전/수치생성전. 자동comparison.json/report.md/index.html 생성준비,원래HR4확률+새2모델만추론하고모든후보공유. 원래R/B4추론반복없음.

### 사용자 test 비교 우선순위 변경 — 2026-10-13 00:10 KST
- 사용자현재학습잠시미루고N3/L2test먼저지시. 실제326 set0fold2 child559685 12,300/16,000step,중간resume상태미저장/마지막에weights·EMA·BN저장하는코드확인. 중도강제종료시약1시간소실이므로controller4026702만SIGSTOP하여다음작업진입방지,현재foldchild는저장까지유지.
- N3/L2test기존대기실행기만교체하여앞선전체EMA/HR완료의존성을사용자승인priority receipt로우회(네트워크우회아님/일정변경). GPU유휴3회검사는유지. 기존학습fold저장후test먼저,그뒤verifiedcontroller SIGCONT 자동재개. 다른사용자작업/가중치/기존child종료없음.
- 우선순위전환실제확인: 326controller4026702 state T(SIGSTOP),현재child559685만12600/16000step계속/14,660MiB. 기존testwaiter796571종료확인. priority wrapper918020/testrunner918030정상생존·오류없음·현재foldGPU해제대기. 종료후test측정완료→CUDAchild종료→4026702 SIGCONT하는wrapper연결완료. HR세트5기존waiter432346의전체EMA완료조건유지. 아직test수치없음.
- test추론6모델완료후CPU결합검증에서停止:推論rawTEAM順(Al/Al3Ni/Primary/Eut)をfreshE MAIN順(Al3Ni/Primary/Al/Eut)へ変換欠落。palette source兩server照合確認。6モデル推論cacheは保全/再実行なし、結合時全parts[1,2,0,3]変換修正。初期failure receipt保全rename、CPUfusionのみretry1。326controller引続SIGSTOP、成功時自動再開。
- N3/L2 test측정완료 2026-10-13 00:42. 6모델×test10단회추론/전환수정후CPU결합만retry. 대조군10마스크exact,가중치·이미지SHA불변. N3 교체35,363/9,074,964px(.389676%),L2 30,921(.340729%),동시58,089(.640102%). 결과+10장확대갤러리report로컬수집/시각검토완료. 10px이상완전재분류기존4연결영역N3 15개최대31px/L2 0/동시26최대33px(정답손실판정아님). 큰조직전반형태유지/주로미세경계및가는조직변화. test GT없음/정확도나최종모델채택주장없음. training_resumed.json 및ps controller4026702 stateS로326자동재개확인. 원래lite16유지/동시교체valid성능미확인.

### 2026-10-13 사용자326계속중단 및동시교체valid지시
- 326controller와현재set1fold1 trainer·그자식DataLoader만SIGSTOP하여메모리/RNG/optimizer상태보존. 자동재개안함,사용자재개지시대기. 다른사용자프로세스종료없음/GPU메모리는정지상태로유지.
- N3→HRs1f2,L2→HRs2f1 동시교체valid20은기존MAIN순v1캐시+lite16freshE기준CPU계산,F재계산. 단독대조/N3/L2/양교체모두보고/GT정확재채점/원본새추론아님/반복valid개발결과. ID codex_lite16_N3_L2_valid_both_20261013.
- 동시교체validCPU검증완료: baseline80.55464329,N3(s1f2교체)80.62272620(+.068083,worst−.084313,17/3),L2(s2f1교체)80.60101979(+.046377,worst−.075833,14/6),양교체80.64316058(+.088517,worstA72−.151261,14/6). 양교체기존worst−.10pp조건미통과/평균이득과국소손실구분/자동채택없음. local summary/perimage/detail/report수집. 326controller4026702/trainer1107074/workers1107646,1107647,1107650,1107651 모두stateT,현재set1fold1로그400step. 사용자요청으로계속정지/자동resume없음/메모리상태유지.

## Codex 전체기존모델 fold대체 감사 + ANA N3 4시드 — 실행 전 등록 (2026-10-13 00:52)
- 사용자승인전체기존모델전수검토및동시에ANA N3 4시드train/valid20. ID codex_allmodel_fold_audit_20261013, codex_N3_seed4_20261013. R329시드팀원작업과중복안함.
- 전수목록: P100 komap_model1/outputs 및원본실험/배포weights, model349_compress와ANA실험의체크포인트/메타데이터/20장예측cache모두조사. 중복가중치·동일예측분리,완료20장cache가있는독립모델전체를lite16에서fold그룹대체/제거에CPU측정. cache없음/입력·class순서불명/미완료는coverage에명시하며평가한것으로주장안함. 원본구조·경로·추론설정확인후필요한미캐시모델GPU평가는새큐에연결. family평균/이미지클래스국소손실/시편제외조건부진단보고.
- N3 4시드: HR336동일부모/원본data_zcdf_png train70/25epoch/512patch scale2 batch4accum2 samples12 lr2e-4 encoder.5 wd1e-4 boundary1. 원래학습소스사용. seed414(재현),415,416,417. valid중학습평가없음/last만. 기존N3seed414가중치는재현대조로보존,새출력seed414와구분. E HR392s1f2 자리만교체/R9/B4wsf1f3/다른HR3/F 고정16개. 단독및최종valid20평균·표본SD·국소손실보고. single-view(v1)추론과기존CPU대조방식동일,별도fresh보정구성은구분. 원본valid/test재학습선택안함/자동채택없음.
- 326paused controller/trainer stateT 및14.6GB메모리확인. 이미완료set0fold1/2보존. 현재set1fold1은400step만진행/재개파일없음. N3에GPU양보하기위해이초기미저장job만종료하고원래로그와재시작receipt보존,326은일시정지계속/나중재개시동일seed set1fold1처음부터(손실약2분)필요. 다른사용자작업종료없음.

- N3 4시드 완료 확인(2026-10-13 02:16 KST): 414/415/416/417 각각25epoch last 및valid20평가 정상완료, controller 종료/queue complete. 최종16개구성 점수80.6282339234/80.6180362434/80.6298614635/80.6351708174%; 평균80.6278256119, 표본SD .0071669424pp, 범위 .0171345740pp, lite16대비평균+.0731823243pp. 모든시드 평균개선/최악이미지 각각−.076428/−.093489/−.099169/−.071630pp. result 입력SHA불변확인, 완료·4개결과·종합결과 로컬수집. 저장확률 E교체+F재계산이며전체fresh검증아님; valid로최고seed선택/릴리스교체없음.

## Codex N3 fresh 검증 → L2 fresh 검증 — 실행 전 등록 (2026-10-13 02:20 KST)
- 사용자 승인 fresh추론검증 및다음L2검증. ID codex_lite16_N3_L2_fresh_valid_20261013. ANA GPU0 idle 확인, N3시드실행종료/326중단유지/HR세트5대기보존/P100 CPU전수감사보존.
- N3는사전에정한재현seed414 새학습last(최고valid seed선택아님), HR392s1f2만교체. L2는원래model_eut_L2/last HR392s2f1만교체. N3/L2동시교체자동채택안함. 먼저N3다음L2단독 검증/학습없음.
- R9 원본valid20이미지부터새추론, E B4ws2fold 및HR3924fold/N3/L2 원본입력부터단회새추론(공통예측공유). E0차이식없음,가족FP16평균및TEAM→MAIN순서정확적용,F구성별재추론. baseline/N3/L2 모두독립GT채점·캐시screen대조·이미지/class최악차이·SHA확인. baseline은80.55464329대조,불일치시원인기록하고자동릴리스교체없음. N3 ZCDF원본20입력/L2 rawRGB,원래HR flip4/신규N3,L2 v1 유지. 기존결과미덮어쓰기.
- 실행확인 02:20 KST: ANA controller1649641/flock1649640, fresh R9 infer_lite subprocess 실행 및322 GPU0 로그진행 확인. 원본R/E/F 가중치·메타SHA preflight통과/20개raw입력SHA 기존manifest일치. 단일실행잠금/기존공통캐시차이식미사용/326자동재개없음. N3고정seed414→L2원래seed361 단계평가,원본실행파일보존.

## Codex lite16 전체 기존후보 오류보완 CPU 검사 — 실행 전 등록 (2026-10-13 02:25 KST)
- 사용자 오류보완 전수탐색 요청에 따라 추가등록. ID codex_lite16_allmodel_complement_20261013. P100 CPU1실행기/torch2threads 추가,기존fold축약 controller2631271 보존/ANA freshGPU검증중복없음. RAM available59GiB 확인. 원본가중치·데이터·릴리스보존.
- 범위: 기존감사519채널검증cache와추가45채널확정cache 총564개저장확률변형(실제모델개수와구분). 정확히동일확률SHA 중복은보완검사한번만. 예측없는가중치·클래스미확인87개는미평가명시,추가GPU추론없음.
- lite16의최종오류 중후보단독정답/현재정답 중후보오답/둘다오답·같은오답·다른오답/GT클래스별·이미지별분해. GT oracle보완량은진단상한이며실행규칙아님.
- 실제적용가능한고정범위: E후보blend .05/.10/.20,각후보F다시계산,R고정. 추가FAl veto는후보비Al·후보max≥.9·margin≥.4·후보label==원래Elabel·원래FAl 선택영역만 Elabel로되돌림. GT를게이트에사용안함. 후보추가시모델수증가/공유여부별도확인;자동채택없음.
- 대조군20최종마스크exact전제. 평균Δ 및최악이미지−.10pp이내 동시조건/최악클래스/개선악화장수·새오류/복구px 보고. 반복valid개발screen/독립일반화추정아님. 원래16개를유지하는축약검사와구분.
- 실행전 채널증명추가감사:45개중25개는상위outputs폴더의다른모델JSON을참조해증명불충분으로제외. 보완검사총539변형=원래519+추가20, 미확인112로정정(564평가로주장안함). 원래축약실행은추가45단계미실행/519계속유지. 보완실행flock2637590 확인/별도단일CPU실행기, 원본결과보존.

## Codex 사용자 ANA 후속 전체 큐 승인 — 2026-10-13 02:32 KST
- 순서: 현재N3/L2 freshvalid완료→L2 4시드train/valid평가→고정N3seed414+L2seed361 단독/양교체valid 및test10변경검토→기존R326 EMA남은6jobs재개/원래평가+lite16평가→기존HR392세트5epoch32~70재개/평가. 사용자명시적재개승인으로앞선326보류해제(앞선검증끝나기전학습시작안함).
- L2 ID codex_L2_seed4_20261013, queue codex_N3_L2_then_resume_queue_20261013. 原model_eut_specialist/train_eut+eut_losses와HR336부모/원본raw train70/12epochs/seed361,362,363,364/LR1e-4 encoder.3 wd1e-4/patch512 scale2 batch4accum2/samples12/eutlovasz weights[1,1.5,1,2]/boundary1/AMP/val-every0/lastonly. seed361원래재현seed고정,valid최고seed선택없음. N3新414와L2新361 test확률만새추론,원래HR/R/B4의이미지SHA검증済test저장예측공유. GT없는test정확도주장없음.
- 326完了set0fold1/2재학습없음,중단set1fold1는resume파일없어0step재시작 필요/400steplog보존별도archive. 원래326평가waiter351745·HRwaiter432346사용/복제안함. 원본가중치·데이터·릴리스 및Claude기록보존. 각선행failure시자동진입중단/수치·완료file확인필수.
- 큐실행연결확인: ANA queue flock1729993 실행/현재fresh validation선행완료 대기,새L2train/test자동단계→원래326run.py의완료set0skip/미저장set1fold1재시작→기존351745lite16평가→기존432346 HR세트5평가까지연결. L2원본raw train/valid images/masks180파일 P100↔ANA SHAexact,원래train_eut+eut_losses import 및실행기syntax preflight통과. N3/L2test원래HR4캐시읽기전용공유(새학습N3/L2만새추론). 원본stop기록보존/새재개receipt는실제326시작시에작성. test자동시각판정/최고seed선택/릴리스교체없음.

- 진행확인 2026-10-13 08:49 KST: fresh R/E/F baseline全20마스크exact80.5546432876,新N3seed414 80.6284614846(+.073818,worst−.076428),原L2fresh80.6009158156(+.046273,worst−.074954). 입력SHA불변/completion수집. L2新seed361~364 모두12epoch完평가完,최종평균80.5969592826/SD .0010396911pp,단독최악−.095358. N3固定+L24시드양교체평균80.6450667713/SD .0043346049,4시드최악−.121948/−.140135/−.147598/−.143500로局所−.10조건모두미통과/양교체채택보류. 新seed414/361test10 control exact,변경N3 .397412%/L2 .353577%/양 .650978%,경계주변99.7366/99.6478/99.5717%,10px이상기존영역완전재분류13(max15px)/0/17(max33px). 원본비교A127/A124/A167각4crop시각확인:주로입자경계와미세조직변경,큰형태급변미관측,전체10시각감사아직未完/test GT없어정확도확정없음. 결과·갤러리로컬수집/원본릴리스유지.
- ANA queue정상326실행으로진입. 326전체8jobs중6완료,현재set3fold1 seed3461 1200/16000step GPU14,634MiB/trainer3695370;마지막set3fold2未開始. 326후lite16평가→HR392set5 원래waiter대기정상. P100519축약변형완료276고유확률오류0/completion·ranking수집. 유일조건통과R329대체후보는existing329결합cache로실제단일모델축약근거확인필요. 보완검사518/539,275고유·243중복·오류0;최고확인후보HR336control E20% 80.62994995(+.075307,worst−.080128),추가가중치구조/fresh및一般化검증未完/자동채택없음.

- 2026-10-13 09:39 완료결과보고수집: P100축약519변형276고유/오류0,보완539변형294고유245중복/오류0 둘다completion완료. 축약唯一pass R329기존3fold결합(FP16weights)임을manifest확인해추가축약기각. 보완최고HR336control4epoch E20%=80.62994995/worst−.080128,17weights필요. E10%=80.60771574/worst−.040811. FAlveto HR347v4+.005353. 공개최고단일모델채택으로확정안함. 112채널미확인+cache없는가중치未評価. 새학습N3/L2test10全10장각4crop갤러리시각확인:大조직급변미관측/微細경계변경/정답추정안함. 종합report experiments/model349_compression/completed_results_20261013/report.md 생성. L2模型単독SD .475265pp(range1.066585),최종결합SD .001040pp와구분해서보고. 326/HR未完維持.

## Codex 새 lite16 N3 구성 채택 등록 — 2026-10-13 09:49 KST
- 사용자명시승인: N3로HR392하나교체한새lite16생성. ID lite16_N3_adopted_20261013. 원래lite16_b413/기존릴리스보존,새버전별도경로/R9 E6 F1 총16. E HR392s1f2→N3新seed414last,HRs1f4/s2f1/s2f4·B4wsf1/f3·R/F그대로. fresh valid80.6284614846/test10새seed414측정済/4seed근거보존.
- E재현: N3전체rawgray/Z±4/CDF를RGB8로round인코딩한뒤원래RGB dataset+single-view scale2;N3개별FP16후family평균FP16,HR392 flip4/widthnorm원래유지. 入力전처리valid/test기존ZCDFPNG와全pixel일치검증후portable코드등록. 새codeconfig/weight manifest/assets overlay 및채택receipt/결과보존,old버전미덮어씀. GPU326/HR큐간섭없음.
- 추가후보N2는ConvNeXtTiny로우선해석해새N3기준의남은HR자리교체CPU검사준비;사용자L2뜻인지async확인중/현재추가모델자동채택안함.
- N3 portable preflight 완료: raw→Z±4/RGB8/CDF encoder valid20+test10全pixelexact,선택16weights 및metadataSHAexact/용량2,424,376,391bytes. 元lite16採択folder不変,新N3 version별도assets overlay/코드. 新N3 Efamily 직접평균CPU대조/F마스크검증예정.
- N2追加screen 실행등록: ID codex_lite16_N3_N2_screen_20261013. ANA CPU2threadsのみ/N3 fresh80.62846148 기준 remainingHRs1f4/s2f1/s2f4각자리1대1대체,3고정후보/모델수16. N2 ConvNeXtTiny v1 MAIN순原保存確率,weight/source확인전fresh나채택주장안함. 원본N3 유지/GTgate없음,입력SHA/20마스크대조、이미지classΔ/새오류·복구px보고. GPU326계속/HRqueue보존.

- N3 사용자채택 완료: lite16_N3_seed414,16weights/validfresh80.62846148,ANA새폴더lite16_N3_adopted_20261013/ADOPTED_BASELINE.json 생성/원래버전보존. ZCDF全30입력exact+E직접가족평균全20確率exact+F全20최종maskexact CPU검증,新portable GPU end-to-end smoke는학습중미실행으로명시. 로컬codezip(가중치미포함)/README/manifest/validation/채택receipt수집.
- 사용자후속N2→L2정정확인. N2 screen child종료/partial결과보존/cancel receipt,추가N2없음. 이미완료L2四시드양교체を新N3基準으로추가계산:seed361/362/363/364 평균Δ+.0124449/+.0169877/+.0145084/+.0224802pp;worst−.1269480(A153)/−.1128679(A63)/−.1213112(A63)/−.1274953(A63),全4−.10미통과. L2추가교체보류/기존326HRqueue불변. 기존326/HR평가는원래lite16대조로계속되고채택N3버전と혼동안함.

## Codex L2 추가교체 최악이미지 비교 웹 — 2026-10-13
- 사용자지시: 새N3lite16→N3+L2 최악이미지만 비교웹띄우기. compare_web/lite16_N3_L2_worst 로컬전용. 기존완료예측만수집/A153(seed361최악)·A63(seed362/363/364최악)/raw·GT·before·after·새오류/복구영역·동기확대 제공. 新GPU추론/학습없음/기존릴리스·큐보존/새모델채택없음. 로컬브라우저미리보기.
- L2최악 비교웹 완료/Chrome 새탭 http://127.0.0.1:8767/ 실제표시確認. 全8시드×이미지쌍 GT독립mIoU가원래기록exact/A153seed361 73.0901→72.9631(−.126948pp),새오류4283/복구2258px. 原/GT/N3before/N3+L2after/오류변화/全클래스변화6패널,동기wheel/drag/손실집중3버튼;손실집중1 클릭AX 3.7배及Screenshot확인. 로컬만,server session48960 유지/기존GPU未干渉.

## Codex lite16-N3 신규기준 전체 보완·내부교체·HR축약 — 실행 전 등록 (2026-10-13 10:22 KST)
- 사용자요청再検査. ID codex_lite16N3_allmodels_20261013. 새채택fresh N3seed414 80.6284614846% 고정. P100 CPU3worker×torch2threads(available48GiB確認),ANA326set3fold2 training/HRset5queue불변. 原lite16대조결과재활용판정아님/N3新error重新검사.
- 후보: 前539검증cache를exact확률SHA로중복제외294후보+새학습N3/L2시드예측8개. class未확인112/cache없는가중치未評価保留。이미검사된source 불변SHA확인/완료파일skip/단일flock.
- 실제가능내부교체: R의9개자리(기본restart61의preaux분포교체는TENT/sourcefresh와구분)、R3262/R3293그룹→후보1、E6개각자리→후보1、E HR3개/seed2pair/B4wspair→후보1(공유시uniqueweights별도계산)。F를다른segcheckpoint로직접load하지않음/F역할의Al판정만candidate로대체하는고정CPU대조별도.
- 보완: E .05/.10/.20 후보추가+F再計算 및후보동의/고확신FAl veto. 모든규칙GT사용안함/GT oracle보완은진단상한. 평균Δ>=0/최악image>=−.10pp/클래스별worst및복구·새오류/시편일관성 후fullfresh優先候補,자동채택없음.
- HR축약우선CPU분해: 남은s1f4/s2f1/s2f4 개별제거(원래group내재정규화) 및모든1개/2개subset,N3직접공유로HR자리에사용するcase、s2pair→1고정평균soup가능성·train70기존HR336/347/N3등구조별분리. HR392同レシピtrain70新대조필요시기존ANA큐後에등록/기존GPUs未干渉. 全体screenはfamily FP16/個別foldcachedprecision한계명시/fresh再評価전release変更없음.

## Codex HR392 동일레시피 train70 직접대조 — 실행 전 등록
- ID codex_HR392_train70_N3_20261013. 사용자의HRfold추가축약/train70대체검토를위한직접실험. 기존저장train70HR336/N3는입력/훈련방식차이가있어서HR392그대로대체가능성분리필요. 原HR392model392/train.py scratch ImageNet cached init/seed1,2 각각70epoch/normalizewidth/patch512scale2batch4accum2workers4samples12lr3e-4encoder1wd1e-4boundary1AMP/rnglegacy/valevery0/lastonly. train70 raw/GT, officialvalid/test학습미사용. modalwidth는train70원래코드정의계산/oldseed2fold1target1230와차이기록(순수데이터수효과만으로주장안함).
- ANA 기존전체큐(R326학습·평가→HRset5학습·평가)의completion後に順次실행/다른프로세스중단없음. seed1/2新valid20 flip4 rawFP32→각remainingHRs1f4/s2f1/s2f4単独대체및s2pair/HR3전체1가중치대체 familyFP16 exact+F재계산. 총16/15/14가중치경우분리/모든후보시드평가/자동최고seed採択없음. 새N3baseline不変/실패시다음진입없음。
- 新N3priority CPU실행기flock2673028/controller2673029,baseline20새E배열/F마스크exact검사통과(NEW_N3_BASELINE20_EXACT),HR removal18종先行進行。後全302候補×25役割screen은追加転送SHAcomplete後に自動開始。ANAtrain70queue flock40248/child40249앞선기존전체큐後대기,ImageNetcached CPUload 성공/targetwidth1278/preflight保存。326最新10:29最後8番目seed3462fold2 5300/16000step/14.6GB정상. 원래controller/evalwaiter/HRwaiter보존。

## Codex lite16-N3 B4 fold 축약 추가 검사 — 실행 전 등록 (2026-10-13 10:36 KST)
- 사용자 B4fold축약 방법 검토. ID codex_lite16N3_B4_subsets_20261013. P100 CPU2threads 단일추가실행기/ANA GPU326·HR·train70queue보존. 기존302×25 내부교체검사는R326/R329그룹및B4ws그룹→既存候補1(기존soup/train70/다른모델)를포함하므로중복실행안함.
- 빠진고정구성만63개: R326현재fold1/2의모든비어있지않은3subset × R329현재fold1/3/4의7subset × E B4ws현재fold1/3의3subset. 新N3/HR3/F・Rroute固定/비율valid재맞춤없음. 전체선택 control新N320PNG exact. subset부분은원래fold별FP16 cache평균(원래fullfamily는fresh原保存平均사용),미세precision한계/fresh検証필요명시. 원본가중치불변/모든63후보/per-image/class loss/복구새오류보고/자동best採択없음.
- 가중치수:現在16−R326제거수−R329제거수−E B4ws제거수,最小12。R322는현재1fold라fold를줄일수없고役割삭제/기존B4공유는별도전수교체수치확인必要。구조상B4ws와R329는別가중치/전처리·훈련path가다르므로동일모델취급안함。
- B4추가대조등록: 原326EMA全学習後既存32個valid概率(4seedsets×2fold×plain/ema/plain_bn/ema_bn)을新N3에서R326의현재2fold전체→각1weight代替、CPU32候補/새推論없음。前lite16_evaluate는paired2weights변동검사라단일weight축약검사와重複아님。ANA GPU큐그대로/원래completion 기다림/16→15임을명시/시드최고선택なし。

- 2026-10-13 10:49 사용자 R329 전체 다른모델 대체탐색 명시. 기존新N3全302후보×25 검사 R_329group_to_one 항목에 이미연결 확인/중복실행없음. R329 fold1/3/4의전체평균source를候補로대체하고기존gamma329[.5,1,0,0]/R/E/F 고정. 단일新model이면16→14,기존사용모델예측공유이면16→13 가능성(실제weight/provenance확인전count확정안함),ensemblecache를1model로취급안함. A71/PrimarySi 및Al3Ni·worstclass중점/GT rule선택없음/자동採択없음. 現2/302완료/errors0、初2候補R329代替低下79.9668/80.0393로pass없음,전체미완료로결론확정안함.
- 사용자제공 R329seed1557/1558/1559 lite19a比較 orig80.5236/new80.3107,80.4626,80.2508、平均80.3414/SD約.109pp/最悪−1.50(A71),−1.13(A186),−3.24(A71)、testchanged.069/.066/.065%。チーム途中結果/独立再採点なし/新lite16-N3と基準区別、남은1560/1561/1562학습중으로사용자이미지에기록됨.

- 2026-10-13 사용자 Downloads lite20_b4ws (1).docx 읽기전용검토/原SHA불변 확인. lite16-N3 교체용그림2구조(R9/E6/F1)·그림3실제R순차교체계수·그림4새seed414test예측2장別生成/PNG시각QA완료/교체자료zip및수정문안TXT·stage/classCSV生成。新valid独立pixel채점R79.79930890→R+E80.59097856→R+E+F80.62846148/E단독79.63024199,상별IoU·정밀/재현·혼동/낮은image갱신。원문경계85.6의정의未確認、新정의4neighboredge+8neighbor1px팽창88.7366%은동일metric로단순교체금지。실제R HR347 tta=False단일view/EHR392 flip4/N3singleview source확인、기존DOCX미수정/GPU큐보존。

## Codex 팀 GitHub R329 시드 결과 확인 — 2026-10-13
- 사용자 요청으로 최신 commit dc9395ec52c9b47bb6886a982d4e6dd0e366a503 읽기전용 확인. 기존 실험/서버 큐 변경 없이 README 및 메타데이터 확인, 팀 평가와 독립 채점 구분.
- GitHub 최신 main commit(11:04:32 KST) 확인결과 B4-329는 fixed seed0 fold1/3/4 원본 3가중치 추가이며 no_seed_search_for_this_component=true, image_miou/standalone_image_miou=null. 다른 브랜치·릴리스 없음. 1557~1562 새 시드 점수는 현재 저장소에 미게시. 전체 기준80.255846/최종80.479068은 R329 고정 상태의 다른 계열 탐색 결과이므로 R329 시드 결과로 해석하지 않음. 메타데이터만 수집/가중치 다운로드·학습·추론·기존 큐 변경 없음.

## Codex R 시작 백본 대조 — 실행 전 등록 2026-10-13
- 사용자 승인. ID codex_R_start_ablation_N3_20261013. 현재 lite16-N3 R 시작 확률만 기존 B4 3그룹(322/326/329) 및 R HR347/원본해상도 ResNet269 저장 확률로 교체. 기존 전체 전수검사와 같은 역할이나 아직 미완료인 현재 구성원에 대한 우선 5개 대조 및 단계별 R/R+E/최종 분석. 새 GPU 추론·학습 없음/CPU 2threads. E/F/결합 순서·계수·밝기/형태 후처리 고정. 현재20마스크exact 대조 확인 후 모든 시작 후보 평가. 입력SHA·이미지별/class 변화·최악 이미지·시편 paired bootstrap 기록. 시작 모델용 TENT 추가·계수 재최적화는 하지 않으므로 고정 레시피에서의 시작 분포 대조로 범위 한정. 기존 작업과 원본 릴리스 보존.
- 사용자 추가 제안: 단독최고78.83은width4000 B4 4fold평균임을문서표·주석으로확인. P100검증cache146 TEAM순서4fold전체valid20 존재. 우선시작대조7종에4fold평균추가. 이어 ResNet시작과B4_4fold시작 양쪽동일예산으로기본4class계수+gamma329Al3Ni/PrimarySi 6좌표 각0/current/1 단회순차탐색(R mIoU목적)실행. HR계수/후처리/E/F고정,전역최적화아님/valid재사용screen/자동채택없음. reopt는첫대조정상완료후CPU순차실행/실패시중단.
- 사용자 base변경최우선/기존작업중지 명시. P100 전체검사parent2673029+workers2673907/08/09+tracker2673906 SIGSTOP확인. ANA 본인queue1729994/326controller1996784/eval351745/HRwaiter432346/train70waiter40249/단일326eval102285 및descendant flock1996783 SIGSTOP, receipt로컬수집. 타사용자작업중지없음/삭제없음/자동재개없음. 고정base7종완료: 현재최종80.62846148, B4_4fold시작최종80.54031518(−.08814631pp), R79.80297725vs기존79.79930890; A181최악−1.288232pp. 보고서78.83 단독확률score78.83342199 재현. 동일예산계수재탐색 reopt2677760 우선진행.

## Codex B4 최고 단독 앙상블 기반 새 R/E/F 설계 — 실행 전 등록 2026-10-13
- 사용자 승인. ID codex_B4base_REF_design_20261013. base=width4000 B4 원래4fold평균(cache146/단독78.83342199 재현) 고정. 기존R 순차route에서출발하지않고확률혼합R 구성. 현재사용가능한R B4gray/B4ZCDF/HR347/ResNetnative/highres및새N3단독 예측을 후보로, 고정전역alpha .1/.25/.5의 전체class·한class 혼합 검사 후 R점수기준 순차최대2모델추가. 동일source재추가금지/GT per-image gating없음/모든후보기록. base4가중치이며한모델로계수하지않음/다중fold donor 실제count별도기록.
- 이후선정R에서 E는현B4ws2+HR/N3그룹을B4비율 .25/.5/.75로비교하고덮어쓰기class조합16개전수, F는기존보정기유지또는미사용을대조. F재학습없음/새OOF라고주장하지않음. 최종평균뿐아니라최악image/class·복구/새오류·모델수·paired시편bootstrap기록,기존lite16N3비교。valid20반복개발screen이며자동채택없음. CPU2threads 별도실행/기존전수·ANA큐중지유지/원본보존.
- 사용자 ANA실험재개 명시(11:34 KST): 저장된중지receipt의실제PID cmdline일치검사후 1996783/1996784/1729994/432346/351745/102285/40249 SIGCONT, 실행/대기상태재확인. 기존순서326학습평가→HR세트5→HRtrain70 유지/단일326CPU평가대기재개. P100기존전수검사STOP유지/base두실험정상진행,새복제실행없음.
- 11:38 KST 새B4base구조 165 R후보+96 E/F조합 완료. 선택R79.44550468%,rawB4base78.83342199 대비+.61208268pp. 최종최고79.67705577%/현재80.62846148 대비−.95140572pp/A71−13.03007199pp/통과0. 최상위E모든class덮어쓰기는R실제로미사용이므로effective7weights(E6+F1)로count정정, R추가의최종효과라고주장금지. 기존R후처리미적용대조라백본열등결론금지. 계수대조19/22진행,현재B4대조최종80.58499737%/ResNet80.64654264%는중간값/채택없음.

## Codex B4 기반 새 R 9가중치 확대 — 실행 전 등록 2026-10-13
- 사용자 승인. ID codex_B4base_R9_20261013. 기존선정새R6(B4원래4fold+N3+ResNet고해상도)79.44550468에서출발, 남은B4gray1/B4ZCDF2/HR3471/ResNetnative1 추가. 실제uniqueweight budget7/8/9별 top2 beam만확장(alpha .1/.25/.5, class단독4종+전체), 순서별중복동일설정재평가방지,전역grid아님/valid20선택screen。R6exact대조후 R7/8/9전체score 및class/worstimage/가중치목록 보존.
- 기존9개R79.79930890와비교할때후처리차이명시. 최상위R9에기존R후처리만붙인paired대조도추가(계수route재사용없음),원래E/F고정참고점수기록하되새E/F설계성공으로주장하지않음. 학습/새GPU추론없음/P100CPU2threads/기존전수STOP유지/ANA재개상태보존/원본릴리스교체없음。
- 계수재탐색정상완료(11:39:42 KST),22개=ResNet/B4각11개 동일좌표순차0/current/1/R점수선택. 두시작모두Al3Ni gamma326 .7→1/Al gamma322 .6→1 선택. ResNet R79.83479572/최종80.64654264(+.01808116pp)/worstA181−.16104217pp,국소−.1조건실패. B4 R79.83059410/최종80.58499737(−.04346412pp)/worstA181−1.13406542pp. 시작B4가단독최고여도고정다른규칙상최종우위없음/전체grid또는새구조전역최적화결론금지/재선택valid자료라일반화증거아님/릴리스미교체. 완료·전체결과·시편bootstrap로컬수집.
- 12:13 결과수집: R9확대181설정정상완료/R6exact. 새R6/7/8/9=79.44550468/79.49561584/79.54656564/79.54481569%, 기존R79.79930890 미달/9번째추가오히려−.00175pp. 기존후처리붙인R9=79.33658491%/고정E,F최종raw79.78303749,post79.71541381/현재최종80.62846148 미달/채택없음. ANA R3268학습32추론640확률및oldlite16paired평가·N3single32후보정상완료/inputSHAunchanged. oldlite16 pairedplain평균80.59888755 SD.01434625/EMA80.59879466 SD.01258830/BN80.59806672 SD.01062055/EMA+BN80.60020715 SD.01159760, 기준80.55464329와구분. newN3단일R326후보최고set0fold2plain(seed3456)=80.68255542(+.05409393pp),worstA71−.42049873pp/32개국소통과0. 국소손실가장작은양수후보set3fold1EMA(seed3461)80.66043116(+.03196967pp),worstA181−.12606484pp/기준−.1미달. 원본릴리스유지/HRset5epoch32-70재개trainchild558319확인.

## Codex seed3461 EMA 교체 후보·이미지별 비교 — 실행 전 등록 2026-10-13
- 사용자 단일EMA대체및이미지별손익 요청. lite15_N3_R326EMA3461_candidate_20261013 별도후보패키지/원래lite16-N3 보존. R3262fold→set3fold1seed3461 ema.pt 하나(SHA074c2bf55886853bcc88d220560ab587770666d5e3986bd8e2067a7f9c49d4d5)/E·F·route계수고정. 이미raw부터추론완료한prob사용해서20마스크CPU재결합·기존32screenexact대조,독립GT채점·원본lite16및현재N3와비교。기존학습큐중복GPU추론없음。valid선택screen/공식독립test점수아님/현재원본릴리스미덮어쓰기。
- seed3461 EMA 후보ANA조립R8+E6+F1=15 SHA검사통과/기존R326fold1slot에서EMA last.pt명으로readonly링크/fold2로드제외/원본릴리스보존. 현재N3 E/F20저장확률CPU재계산마스크GT독립채점및원래32screen전20일치/inputSHAunchanged. 현재N3대비80.62846148→80.66043116(+.03196967pp),12개선8악화/복구9972·새오류9105,시편bootstrap95[−.02358024,+.11018318]pp/이득A71+.65593123pp·최악A181−.12606484pp(특히Eut−.67114412pp). N3전원본lite16 80.55464329와비교도별도CSV보존. 새로운15개후보풀GPUfresh는아직미실행/원본모델활성릴리스미덮어씀.

## Codex B4-base 고정 R의 새 E 구성원·새 F 학습 — 실행 전 등록 2026-10-13
- 사용자 명시요청. ID codex_B4R_newEF_20261013. 대상은새로쌓은R79.5448이아니라기존route재탐색 B4_4fold R79.83059410이다. reopt_result B4_4fold config고정/기존E,F대조최종80.58499737 exact 재현,현재ResNet-base80.62846148도비교. P100 CPU2threads 각각E실행기+F학습기,기존전수STOP/ANA HR세트5평가큐보존/GPU작업없음.
- 새E는검증된저장확률후보에서기존보완검사상위12개및축약상위12개+새N3/L2시드8개+원래B4/HR/R구성원 등중복제거한우선후보로구성원교체·추가. standaloneE/HR3공유대체/B4pair대체/HR각slot대체/기존E에신규25%혼합을고정된두class선택(Al3Ni+Al 또는all)로screen후상위및국소손실우선후보에모든class조합16개와새Fgate평가. 전체기존302전수라고주장하지않음/실제ensemble캐시가중치count확정전1모델취급금지.
- 새F는기존CNN재사용이아니라B4 held-fold train70 OOF logprob4+3x3주변평균prob4+원본Z밝기/경사/엔트로피/확률차이의12특징으로학습하는4class 선형logit보정기2종(일반CE·상불균형가중CE). valid20학습미사용,train70 random6000px/image/12epoch/AdamW/lastonly/seed819349, CPU훈련. B4valid4fold평균 vs trainOOF단일heldfold 분포차이명시,새R전체OOF로학습한F라고주장하지않음. 학습archiveSHA·heldfold출처·GT/inputSHA 기록。
- 선택E와새F결합은F단독확률confidence .6/.8/.9/모든픽셀vsR,E불일치영역/덮어쓸class(Al,Al3Ni+Al,Eut,all) 고정grid;규칙에GT이미지별선택없음。성능/최악image-class·새오류/복구·모델비용/시편bootstrap보고/valid개발screen/원본릴리스자동교체없음。
- 실행확인: 새F train70 heldfold archive SHA/입력·GT SHA검사통과,420000학습픽셀/12epoch×plain·balanced2종완료/valid20학습미사용,checkpoint·receipt로컬수집. E검색실행기2683088,원래currentN3대조마스크exact후B4R79.83059410·기존E/F80.58499737 재현통과,중복제외실제E후보33개/첫후보16설정+대조2완료. 전체302전수아닌우선33개탐색/기존전수STOP·ANA원래실험보존。
- 사용자 A71과test유사입력 확인: test10원본전체와A71원본시각검토 및밝기CDF거리/표준화gradient·LBP질감 JS순위계산,원본SHA보존. A38밝기분포·대비근접(A71평균160.419/SD66.116,A38평균165.500/SD66.331),A167평균159.249/질감근접,A122질감지표근접이나평균179.002/대비46.284로다름,A51평균159.567이나큰검은다각형영역차이. descriptor동일크기resize탐색용으로학습embedding·유사확률·test성능주장없음. A71GT PrimarySi없음과test상유무동일판정불가/GT없어A71성능저하가test로전이한다고결론금지. 비교PNG·전체11장contactsheet·JSON저장.

## Codex 사용자 일반 가중치 교체 채택 — 실행 전 등록 2026-10-13
- 사용자 seed3456 일반(last.pt) 후보채택 명시, 현재기준 lite15_N3_R326plain3456/총15(R8+E6+F1)로변경. 기존R326fold1·2평균→새seed3456fold2하나, N3seed414/E/F/route/후처리고정. 원본lite16·lite16-N3·EMA후보 및기존릴리스보존,새별도채택폴더 lite15_N3_R326plain3456_adopted_20261013. SHA06602d7378249de97a0055923861c70d053b7fcbb374597dbd6187e9688605c6 확인/입력SHA·manifest15개확인/저장fresh개별확률CPU재결합20마스크독립GT채점·screenexact후채택receipt생성。최종80.68255542%/+ .05409393pp/국소A71−.42049873pp 사용자비교후선택/기준위반으로재승인요청안함. 현재pack全GPU fresh검증은완료후ANA기존큐를보존한후속validation으로등록. 진행중B4newE/F 비교기준은실험동결80.62846148그대로이며향후현채택80.68255542대조추가.
- 일반seed3456 사용자채택완료: lite15_N3_R326plain3456/15weights(R8 E6 F1),실물last.pt SHA06602d...검증/메타15manifest정확. valid20CPU재구성20장screenexact/inputSHA불변/최종80.6825554175(+.0540939329ppvsN3),원본lite16 80.5546432876도별도보존. ANA새adopt폴더ADOPTED_BASELINE.json및K/CURRENT_ADOPTED_MODEL.json 생성/로컬manifest·마스크·코드zip·현재모델pointer수집,원래모델·EMA후보·기존릴리스미덮어쓰기. 新infer_compact의R-release재사용은fold_ids동일검사추가로원래2fold출력오사용차단. 全GPUfresh valid20→test10 실행기등록/기존HRtrain70completion후 GPU0안정idle확인후실행/partialoutput발견시재실행금지/다른작업종료없음。새B4R E/F실험은동결기준 유지하고향후현재채택baseline비교별도추가。

## Codex R326 fold1 EMA vs fold2 일반 test10 우선 비교 — 실행 전 등록 2026-10-13
- 사용자 두후보test예측비교명시. ID codex_R326_plain_EMA_test10_20261013. seed3461fold1ema SHA074c2bf... vs seed3456fold2plain SHA06602d... 각각1R326/총15weights,기존lite16-N3 control 보존. 공통R캐시ANAadd/base_test(pairwise고정) 및기존N3test-E공통prob 재사용전 control10 최종maskexact·입력SHA 확인. 해당2R326 testprob 아직없음만원본부터GPU0 D4추론/FP16평가quantization동일적용/원본데이터가중치보존。
- HRtrain70현재seed1중(약epoch47): 부모controller40249만진입후SIGSTOP하여현재training child765784는70epoch까지계속/손실없이학습종료GPU반환후비교우선실행。비교정상/실패후부모SIGCONT복귀를finally보장,기존freshqueue는train70완료대기보존/다른사용자작업중지없음。
- GT없으므로정확도판정안함. 두후보간/원본N3간 changedpixel/class전이·class면적·변경connectedregion·근경계비율,원본+control+EMA+plain+변경표시 cropboards생성. A38/A167 우선시각검토및test10전체검토/통계·마스크·입력SHA수집/일반후보채택상태자동변경없음。
- test 비교실행확인: controller965681,원본SHA·Rcontrol10·N3finalcontrol10 exact검사모두통과. HR부모40249 STOP/child765784 학습계속확인/현재seed1완료후두R326GPUtest추론우선. 기존HRrawtestcache는codex_lite16_N3_L2_test_20261012에보존됨/새N3prob는codex_N3_L2_seed_retrained_test_20261013 것 사용하여새N3controlexact 보장. 새GPU추론2후보만/부모finallyresume등록/일반모델현재채택유지.
- 13:11:11 KST 새B4R E/F 탐색정상완료: R79.83059410고정/새E33개구성원후보530설정+class재검사128/새OOF F2종결합294설정. 최고80.61122311%, 기존B4R+oldEF80.58499737 대비+.02622575pp; 현재채택lite15plain80.68255542보다−.07133231pp。새E=기존E75%+HR336control4ep후보422 25%,Al3Ni/Al교체; 새Fbalanced확률≥.6 Al만반영/F혼합전80.60196529→+.00925782pp。oldN3기준worstA181−1.02236304pp/현재15기준국소차이별도수집/채택없음. 원본릴리스보존/기존전수STOP유지. 完료·모든候補/ perimage 로컬수집。

## Codex lite15 기준 B4·HR 축약 재검사 및 채널 오류 수정 — 2026-10-13
- 사용자 B4width4000 3fold 및EHR추가축약방안검토. sourceANAadd/probe_r_b4.py는이미TEAM_TO_MAIN변환한MAIN순서cache를저장함이원문확인됨. 기존newscan/init rfolds에서[[1,2,0,3]]를또적용해중복변환한오류발견. 기존63개B4subset의partialR결과 및새302×25의개별Rfold교체역할만무효표시(원본결과보존),all-fullgroup control·전체R그룹교체·EMA/plain단일R326군전체교체·HR/N3/E/F새base대조 unaffected. oldR326 one-fold-only(기존반대fold와섞음)결과도동일사유제외,양fold재학습mean평가정상. 현재선택seed3456plain score80.68255542는별도후보전체R326교체경로+독립20마스크exact이므로불변。
- 새ID codex_lite15_B4_HR_reduction_20261013. 현재seed3456plain R326 FP16 probabilities20을P100복사/신규BASE80.68255542 exact확인후R329의nonempty7subset(MAIN순서그대로),남은HR3의remove/share18종및R3297×HRsubset8=56joint CPU검사. 구성별mean/worstimage/class 실제uniqueweightcount기록·각후보F재계산/자동채택없음。P100既存scanSTOP유지/ANA test우선job·현재학습보존/원본가중치보존。

## Codex 사용자 전체 축약 방향 실행 승인 — 2026-10-13
- 사용자 우선순위 및후속모든방향 진행명시. 순서①현재15 기준B4/HR 제거67검사 완료②기존HRtrain70(seed1/2)새15기준 대체평가③원래R329fold1/3/4 각각train70저학습률추가학습+plain/EMA/SWA 검증④기존모델공유/N3공유 및⑤낮은우선순위fold soup도 별도호환성·BN검사후검증. 기존원본데이터·가중치·릴리스·Claude기록보존/자동최고시드채택없음/타사용자작업중지없음。
- 新ID codex_lite15_B4_train70_EMA_SWA_20261013. ANA기존HRtrain70 및채택15 freshvalid/test 완료後 GPU0순차. 原R329fold1/3/4각완전checkpoint에서fulltrain70 시작/1680step×batch2=3360randompatch(균일표본 기대48/image),targetwidth4000/crop1536/ZCDF/image_global/uniform/원래CE+Dice/LR3e-5encoder.2/wd1e-4/AMP/workers4/BN통계고정/seed819329+fold。EMA.995부터840step/SWA840..1680매120step8snapshot/한학습에서3가중치저장/val학습및epoch선택없음。9valid候補를R3293→1(총13),E_B4ws2→1(총14),양쪽동일checkpoint공유(총11)로각평가,실제uniquecount·worstclass·test변화조사。
- fold soup方向: 원래3fold의모든2foldpair 및3fold평균4종별도생성,키·shape·dtype·전처리호환성검사,BN 원래평균 vs train70 재계산200batch 대조8valid候補/서로다른basin위험으로단독성능기각도유효결과/原파일덮어쓰기없음。共通현재15baseline exact20전제。
- 모델공유CPU는R329전체를현재R326/R322/R347/R269/N3로공유 및제거대조, E_B4ws각fold/전체공유·HR다중slot공유 중현재67검사에없는조합만후속수행/원래67의N3단일/다중 및R347/269단일HRshare중복안함。
- 후속GPU구현확인추가: 원래infer_lite R B4 worker는tta=False실제singleview,문서config b4_tta=d4필드는실제미사용. 新326全32probはrun.py predict_probs(tta=True) D4로생성됨. 현재일반/EMA候補 채점은D4로일관돼유효하나원래2fold singleview와비교에는TTA차이가포함되어순수시드효과로해석금지. 新15패키지Roverlay에 b4_tta_by_component={322:false,326:true,329:false}를실제호출에적용/元부모미변경/현재R_releasemanifest추론메타교정/同TTA검사추가。전체fresh큐시작전수정확인/채택80.6826 cachedscore不変/가중치감소가추론속도감소를뜻하지않음。앞으로warm329 R는원래singleview/EはD4 각각정확히평가。

## Codex 우선 축소검사·test10 후보 비교 진행 — 2026-10-13 14:10 KST
- `codex_R326_plain_EMA_test10_20261013` 완료. 원본 test 입력 SHA·R control·N3 최종 control 각각 10/10 exact, 입력 SHA 유지. EMA3461과 plain3456의 15가중치 예측 변경 7,590/9,074,964px (0.083637%). A51 3,749px(0.3227%)가 최대, 다음 A124 1,244px, A167 831px. 변화의 대부분은 기존 경계 2px 이내(7,451px); 정답 없는 test라 정확도·우열 주장은 안 함. 마스크·이미지별 JSON·그림 완료. HR train70 controller는 SIGCONT되어 seed2 학습 재개/현재 epoch68/70.
- `codex_lite15_B4_HR_reduction_20261013` P100 CPU-only 67/67 완료. current15 exact20 및 원본 R per-fold MAIN class-order·FP16 평균 재현 검증 통과. baseline 80.68255542. R329 fold 3→2는 유지 {1,3} 80.62727013 (−.055285pp, 최악 A186 −1.2051), {1,4} 80.62257698 (−.059978pp, 최악 A11_step1 −.7049), {3,4} 80.38901187 (−.293544pp, 최악 A71 −5.7147): 단순 fold 축약은 보류.
- E HR 후보 중 s1f4만 남기고 seed2 f1/f4를 N3와 그룹 교체한 13 weight 조합은 80.69860745 (+.016052pp), 13/20 개선·7악화, 최악 A153 −.222944pp. N3를 s2f1 단일슬롯에 공유하는 14 weights 조합 80.70142097 (+.018866pp), 최악 A153 −.268842pp. 기존 15 weights 유지 대비 평균은 소폭 상승하지만 국소손실 −.1pp 기준을 통과한 축소안은 0개라 채택하지 않음. HR 3개를 모두 제외하고 N3만으로 두 25% HR 그룹을 채운 조합은 12 weights, 80.52134600 (−.161209pp). 전체 ranking·per-image 수집경로 `experiments/model349_compression/codex_lite15_B4_HR_reduction_20261013/collected/`.
- ANA `codex_HR392_train70_N3_20261013` seed2 train70은 14:10 확인 시 epoch68/70, GPU0 사용중. P100 GPU3은 다른 소유자 PID 3557307의 13GB 메모리 점유가 확인되어 건드리지 않음. fresh15 queue 870152 및 후속 model-sharing 2689248은 기존 의존성 완료를 기다림. 이후 R329 train70 EMA/SWA와 fold soup는 사전 등록 범위대로 순차 진행하며, 실제 GPU가 비면 시작 전 상태를 다시 확인한다.

## Codex 축소 파이프라인 후속 — 2026-10-13 14:31 KST
- HR392 train70 seed1·2 70epoch/valid 추론 모두 완료. 저장확률을 현재 lite15 경로에 FP16 배포 정밀도로 다시 끼운 CPU 재평가 10개 중 seed1의 s2f1 교체만 사전 기준 통과: 80.70160649% (+.01905107pp), 최악 A153 −.09179493pp, 12/8 이미지 개선/악화. seed2 s2f1은 +.01531906pp이나 최악 A153 −.12790890pp라 미통과. 다음 test followup은 seed1 후보만 예측 변경량을 측정하며 test 정확도는 주장하지 않는다. 결과 `experiments/model349_compression/codex_lite15_HRtrain70_evaluation_20261013/collected/`.
- lite15 전체 fresh 재추론은 초기 실행 중 R 경로 6개와 valid20 R 마스크가 완성됐고, R 결과가 기존 채택 R 마스크와 20/20 픽셀 exact였다. E 진입에서 두 코드 의존성(재현 코드·자산 manifest)의 wrapper 경로 누락으로 중단. 원본 실패 로그와 부분 결과는 보존, 경로를 package-relative `runtime/`로 고치고 해당 원본 manifest SHA 두 개와 코드 SHA 검증. 중복 R 실행 없이 E valid 원본 추론을 복구 중이며 현재 GPU0에서 유효 출력이 생성되고 있다. fresh valid/test 최종 완료 전까지 다음 모델 학습을 시작하지 않는다.
- 재시도 연결 중 발생한 이전 대기기/컨트롤러의 실패 파일은 별도 `failure_attempt*`로 보존. HR CPU 재채점 완료 및 import 경로 복구 중이고, fresh_completion 생성 후 HR test followup→B4 warmstart/EMA/SWA 및 soup→model-sharing 순으로 큐를 다시 연결한다. 모델학습 가중치·원본 릴리스는 변경하지 않았다.

## Codex lite15 fresh 재현 완료·후속 큐 재연결 — 2026-10-13 14:51 KST
- 복구된 lite15 fresh source inference 완료: R 6개 component는 기존 확인한 결과를 재사용했고, E source valid20 추론·결합 후 full source test10 추론까지 끝남. fresh valid image-mIoU **80.6825554175%**, 저장된 adopted valid20 출력과 20/20장 픽셀 exact(차이 0). 입력 30장 SHA와 final mask SHA valid20/test10 각 manifest 대조 완료. test GT 미사용·정확도 미주장. import-path 이전 실패 receipt는 `fresh_failure_before_import_path_fix.json`로 보존됨. local 수집 경로 `experiments/model349_compression/lite15_N3_R326plain3456_adopted_20261013/` (fresh recovery JSON, manifests, final labels, 로그).
- HR seed1/s2f1 train70 후보만 test followup 재개(PID 1539741; GPU0), B4 train70 EMA/SWA·fold-soup runner 재개(PID 1539742, HR test 완료 대기), B4 test followup waiter 재개(PID 1539743). 과거 failure.json은 지우지 않고 `failure_attempt1_archived_after_fresh_recovery.json`로 보존. HR test 결과는 test mask 변화량만 기록하고 GT/정확도 주장하지 않음.
- P100 CPU model-sharing runner PID 2692046 실제 생존 확인, 아직 warm-phase 준비 marker를 기다림. P100 GPU3 등 다른 사용자 프로세스 미중단. B4 3fold train70 warmstart/plain·EMA·SWA 9안 + fold soup 8안 valid 평가 완료 후 P100 공유검사가 자동 이어지도록 대기 유지. 기존 가중치·채택 release 변경 없음.
- HR test followup 완료: seed1 train70 s2f1만 비교, baseline 대비 changed pixels 18,122/9,074,964 (0.199692%, 이미지별 최대 A124 3,348px), 입력 SHA 10/10 exact, GT 미사용. 변화량에 따른 test 정확도 주장은 없음. B4 train70 warmstart 1/3/4 및 EMA/SWA/soup runner가 GPU0에서 fold1 학습을 시작(PID 1562163; 1,680 optimizer steps), B4 test waiter는 완료까지 대기. P100 model-sharing waiter는 계속 대기.
- 15:02 현황: B4 warmstart fold1은 1,679 optimizer updates/1 skipped update, 8 SWA snapshots, 839 EMA updates, BN frozen으로 정상완료. fold3(목록상의 두 번째 학습; PID 1617547)가 GPU0에서 시작, fold4는 대기. 각 학습 1,680회 후 plain/EMA/SWA 3종을 보존·valid 채점. P100 공유 runner PID 2692046은 warm/soup marker 대기 중.

## Codex lite15 기준 전역 모델 탐색 재개 — 2026-10-13
- 사용자가 기존 P100 전역 탐색을 lite15 기준으로 다시 돌리도록 지시. 이전 lite16 global review는 302개 중 15개에서 정지돼 있었고, 당시 `progress_before_lite15_restart_20261013.json` 및 `status_before_lite15_restart_20261013.json`을 기존 폴더에 보존한 뒤 우리 소유 SIGSTOP 프로세스만 종료했다. 기존 후보 확률·로그는 보존하며 lite16 점수를 lite15 결과로 재사용하지 않는다.
- 새 ID `codex_lite15_allmodels_global_20261013`, P100 `/workspace/model349_compress/codex_lite15_allmodels_global_20261013`. lite15 fresh valid20 전수 출력 기준 사전검증 통과: R fresh exact20, E 재구성 exact20, 최종 마스크 exact20, valid 입력 SHA 불변, test GT 미사용. 기준 valid 80.6825554175%. R326 plain3456 FP16 확률 아카이브 SHA256 `8598bb1b12c2444dbbbfeb68fbb71110e94c332f930de75ffbce5d45ac4f3a2b` 확인.
- 조사 범위 302개 후보/23개 규칙, P100 CPU 전용(3 worker×2 torch threads). 우선 lite15 HR 축소 집중 검사를 수행한 뒤 후보 확률 coverage 및 전체 탐색으로 이어짐. 15:25 KST 확인 시 PID 2697063 실행 중, 집중 검사 단계 CPU 사용 중이며 `progress.json`/전역 coverage는 아직 생성 전. P100 GPU는 우리 탐색에서 사용하지 않음; 타 사용자 GPU 프로세스는 건드리지 않음. 모델-sharing 대기 프로세스는 이 탐색 종료 전까지 정지 유지.
- 동시에 ANA B4 train70 warmstart runner는 fold1/3/4 학습을 끝내고 9 plain/EMA/SWA valid 후보 평가 단계이며 2/9 모델 처리 완료(15:23 KST 확인). 예전 fresh/HR 실패 기록은 stale failure 파일로 보존돼 있으나 실제 fresh completion, HR valid completion, HR test-followup completion이 존재하고 현재 평가가 진행 중이므로 중복 학습하지 않는다. 기존 릴리스·원본 가중치는 미변경.
- 15:28 KST 업데이트: 집중 HR 축소검사 18개 구성이 완료되어 `HR_removal.json`과 302후보 `coverage.json`을 로컬로 수집. 최선의 기준 비교는 N3를 HR seed2 fold1 슬롯 하나에 공유(14 weights) 80.70142097%, +.01886555pp이나 최악 이미지 A153 −.268842pp라 사전 국소손실 허용치(−.1pp)를 통과하지 못함. 다른 단일 HR 제거·N3/R347/R269 공유도 통과안 없음. 전체 scan으로 단계 전환, 302 후보 coverage 기록됨. P100 controller 2697063 및 pool worker 2697833–35 실행, GPU 사용 없음 확인. 아직 `progress.json` 카운트 첫 결과 대기 중.
- 15:31 KST 상태: P100 전역탐색 worker 3개가 lite15 구성 및 모델확률을 각자 초기화하고 있음(각 RSS 약 6.3–6.5GB, CPU 약 120%씩). 아직 첫 후보 완료 전이라 `progress.json`이 없으며 controller와 worker 모두 생존. 가용 RAM 34GB, P100 GPU 우리 job 0%; 메모리/진행을 계속 감시하고 있음. ANA B4 valid 후보는 6/9까지 진행.
- 15:38 KST 상태: ANA B4 plain/EMA/SWA valid 평가 8/9. P100 3 workers는 source cache bulk-load 중으로 각 RSS 약 6.4–6.7GB, CPU 약 130%이며 누적 읽기량이 증가. 후보 첫 저장 전으로 progress 카운터가 아직 비어 있음(약 5분 worker 초기화). 프로세스 생존, failure 없음, GPU 미사용, RAM available 34GB 확인. 후보 결과가 기록될 때까지 실행 중단/재시작하지 않고 추적.
- 15:40 KST warmstart 후속에서 fold-soup BN 보정이 args metadata에 crop이 빠져 실패했음. 학습 인자 원본 `training/from_fold1/args.json`과 실제 레시피 확인: crop=1536. `calibrate_soup()`에서 이를 명시하도록 로컬/ANA 실행 코드를 수정하고 실패 기록을 `failure_soup_recalibration_missing_crop_archived.json`로 보존. 재개 PID 1851062, 기존 fold 학습·9개 valid 평가·soup 가중치 생성 완료 산출물은 건너뛰고 BN 보정 단계만 재개 중. 다른 사용자 작업 없고 ANA GPU0 비어 있는 것 확인.
- 15:37 KST 진행: ANA soup_13/14/34 BN recalibration 완료, soup_134 진행 중. GPU0은 본인 보정 작업 외 사용 확인되지 않았고 점유 약 577MiB, 사용률 0% 순간 샘플. P100은 후보 cache 0–2/302, 오류 0. 세 worker가 다음 archive 후보(각 20개 valid 확률)를 CPU 처리 중이며, 13분 구간 동안 3개 완료로 처리 속도가 느려 보이나 모두 CPU 사용 중이고 서버 RAM 여유 34GB 수준에서 안정. 중단/재시작하지 않음.
- 15:41 KST ANA 재개: crop=1536 보정 수정 후 fold-soup 네 조합(soup_13/14/34/134)의 train70-only 200-batch BN recalibration을 모두 완료. 이어 original BN vs recalibrated BN 8개(valid20) 채점을 시작했고, 확인 시 ANA GPU0 사용률 99%, 메모리 약 7.2GB로 우리 B4 평가가 실행 중. 9개 warm 후보 평가는 캐시로 건너뛰며 학습 재실행 없음.
- 15:45 KST 결과 수집: B4 warmstart plain/EMA/SWA 9개 후보 valid20 채점 완료(각 후보가 R329 교체 13weights, E B4ws 2개→1개 14weights, R/E 공유 축소 11weights의 세 역할, 총 27 case). 기준 lite15 80.68255542% 대비 전 후보가 낮음. 각 역할 최고는 E축소 fold3 plain 80.62782374% (−.05473167pp; 최악 A153 −.686767pp, 최악 클래스 −1.899814pp, 7/13 이미지 개선/악화), R축소 fold1 SWA 80.45546319% (−.22709223pp; 최악 A71 −1.808054pp, 최악 클래스 −3.628454pp), 동시공유 fold1 SWA 80.37912766% (−.30342776pp; 최악 A71 −1.798486pp, 최악 클래스 −3.638001pp). 어떤 후보도 대체 채택 기준을 통과하지 못함.
- 세 train70 trajectory(329 fold1/3/4 부모) 모두 1,680 update 완료; fold1은 1,679 update+1 skipped, fold3/4 1,680+0 skipped. 각각 EMA 839/840/840 update, SWA snapshot 8개, 약 500초. plain/EMA/SWA 중 E 교체 최고는 fold3 plain으로 EMA/SWA도 소폭 더 낮음. 현재 BN recalibrated soup 4개가 valid20 재채점 중; test10 followup/모든 soup score/전체 completion은 미완료. warm 결과 로컬 수집: `experiments/model349_compression/codex_lite15_B4_train70_EMA_SWA_20261013/collected/warm_phase_ranking.json`, per-image 상세 JSON.
- 15:43 KST: P100 full lite15 scan 6/302, errors 0 (cache IDs 0–2, 9, 12, 13); 3 workers active, GPU unused. ANA는 soup_13 original-BN valid predictions 3 roles 완료 후 soup_13 recalibrated-BN probabilities/evaluation 진행 중. Warm-phase 결과는 이미 수집·기록했고 soup 최종 결과 기다리는 중.
- 15:45 KST: P100 lite15 전역 scan 9/302, 오류 0. 세 worker의 처리량은 계속 확인 중이며 기존 결과 캐시/입력은 보존. ANA fold soup 검증은 soup_13과 soup_14(원본BN/재보정BN), soup_34 originalBN의 probability와 최종 valid mask 3역할이 생성됐고, soup_34 recalibratedBN 진행 중. 점수표는 8후보 전체 종료 후 기록.
- 15:52 KST ANA B4 warmstart/soup 전체 완료: 17개 checkpoint×3 역할=51 valid case, baseline mask exact20, 원본 입력/checkpoint SHA 불변, release 변경 없음. 51개 중 사전 통과(mean≥baseline 및 worst image≥−0.1pp) 0개. Fold-soup 최고는 E B4쌍→1의 soup14 BN재보정 80.51379654%(−.168759pp, worst A181 −.4623pp), R329→1은 soup13 원본BN 80.42069630%(−.261859pp, worst A186 −1.9142pp), R/E 동시 11 weights는 soup14 재보정BN 80.18216003%(−.500395pp, worst A71 −3.8778pp). warm 후보 최고는 기존 기록과 같이 E 역할 fold3 plain 80.62782374%. Test10 followup은 51 valid case 중 기준 통과가 없어 예측을 생략했으며 `GT_test_used:false`; 정확도 미주장. completion 및 test skip receipt 로컬 수집. P100 full global scan은 동시 12/302, 오류 0.
- 15:54 KST P100 속도 조정: host 48 cores, load≈10–11, available RAM≈31GB 확인 후 3→5 workers로 늘리기로 함. 기존 progress/status를 `progress_before_worker_rebalance_20261013.json`, `status_before_worker_rebalance_20261013.json`로 보존하고, result.json이 모두 확정된 12/302 시점에서 우리 controller/3 workers만 SIGTERM. 현재 실행 중이던 cache 26–28 결과 파일은 없음을 확인; 이미 확정된 cache 0–25 결과는 그대로 재사용. 종료 시 multiprocessing semaphore 정리 경고 외 failure 없음. local/remote review.py에 pool=5 반영·py_compile 통과, 새 controller 2700407/Python2700409 시작. preflight 중, GPU 미사용, available RAM 53GB. 타 GPU 작업 건드리지 않음.

## Codex HR392 train70 두 자리 대체 보류 및 seed3·4·5 확대 — 2026-10-13
- 사용자 승인: seed1의 E HR392 seed2 fold1·4 두 슬롯을 동일 train70 HRNet 하나로 대체하는 14weights 안을 보류 후보로 등록하고, seed3/4/5 세 개를 추가 학습해 기존 seed1/2 포함 5seed 편차 검사. 반복 요청은 같은 실험 승인으로 처리하며 중복 job 없음.
- ID `codex_HR392_train70_seed345_lite15_20261013`. ANA GPU0 실제 idle(0%,15MiB), 기존 seed1/2 completion 확인. 원래 HR392 train70과 동일 ImageNet 초기화·70epoch·마지막 checkpoint·patch512/inputscale2/widthnorm/samples12/batch4/accum2/workers4/LR3e-4/encoder_mult1/wd1e-4/boundary1/legacy RNG/val-every0. 새 seed3,4,5만 순차 학습·flip4 valid 추론. 기존 checkpoint와 release 보존.
- 주 평가: 고정 lite15 R/E/F에서 E의 s2f1+s2f4 두 슬롯→동일 새 checkpoint(14weights). 대조: s2f1만 교체(15weights). 결합 비율·F 고정, final valid20 및 이미지/클래스별 delta 재계산, baseline exact20 선행. 기존 seed1/2 확률 재사용, 전 시드 평균/표본SD/범위 및 이미지별 seed 변동과 worst loss 보고. best seed 자동채택/valid epoch 선택/시드 앙상블 없음. 기존 −.1pp gate 유지; seed1 두자리 후보 80.69353347%(+.01097805pp), worst A181−.12061796pp를 보류로 명시. 공식 valid는 개발 점수이며 test GT 미사용.
- 실행 확인: ANA controller flock2116138/Python2116139, seed3 training child2122925. GPU0 96%/6.5GB, epoch1 step100/210 확인. seed3→4→5 자동 순차 학습/각 valid inference 및 고정 2role 평가, 완료 시 5seed summary 생성.
- preflight는 adopted lite15 최종 마스크 exact20 통과, 원래 seed1/2 checkpoint SHA를 각 학습 completion과 대조 통과. 단일슬롯 후보 replay는 기존 per-image 값 대비 최대 .001256996pp 차이가 있었으므로 기존 점수와의 차이를 별도 JSON에 보존하고 5seed 모두 같은 코드로 재채점. 주 평가 두 슬롯 대체의 seed1 80.6935334678%, seed2 80.6768392128%는 기존 저장 점수와 일치. 학습 전 preflight 실패 3건은 원본 로그/JSON에 보존, 학습 중복 실행 없음. `prior_seed12_score_comparison.json`, `preflight.json` 확인. 릴리스 변경 없음.
- 최근 서버 상태 보고: P100 global scan 42/302(13.9%), errors0, controller2700409 생존, 가용RAM11GB. ANA 추가 HRtrain70 seed3 child2122925 epoch23/70 step100/210, 최근epoch35.9초, GPU0 94%/6533MiB. seed4/5는 같은 controller 뒤에 순차 대기. 새 seed valid score는 아직 없음; loss로 채택 여부 판단하지 않음. 기존15/보류14 구성 미변경.

## Codex HR392 train70 5seed 검증 완료 — 2026-10-13 20:44 KST
- ANA seed3/4/5 70epoch 학습·flip4 valid20 추론·고정 lite15 재채점 모두 정상완료, GPU0 idle(0%,15MiB). completion: original inputs SHA unchanged, release_replaced false, test GT unused. seed1/2 재사용 포함 전체5seed 결과 로컬 `codex_HR392_train70_seed345_lite15_20261013/collected/`에 ranking/per_image/seed_variability/completion/scoring_receipt 수집.
- 주 비교 E HR392 seed2 f1+f4 두슬롯을 동일 train70 HRNet 한개로 공유(총14weights), lite15 baseline80.68255542 대비: seed1 80.69353347(+.01097805pp), worstA181−.12061796; seed2 80.67683921(−.00571620pp), worstA67−.26498956; seed3 80.69337432(+.01081890pp), worstA153−.37796128; seed4 80.69674872(+.01419330pp), worstA153−.09519538; seed5 80.72075530(+.03819989pp), worstA63−.05187782. gate(mean≥baseline/worstimage≥−.1pp) seed4·5 통과, 나머지 미통과. 기존 gate 변경 없음. seed5는14/6 이미지 개선/악화, 최악 클래스−.15166128pp.
- 두슬롯5seed 평균80.69625020(+.01369479pp), 표본SD .01575743pp, range .04391609pp. 전체평균편차보다 국소변동 큼: A67 range .43370182pp, A153 .38376299pp. 따라서 신규 축소 후보는 생겼지만 학습 recipe가 모든seed에서 안정하다고 주장하지 않음. bestseed valid 선택의 편향 명시, 자동채택 없음;14weight test10 변화 비교는 아직 미실행.
- 한슬롯15weight 대조는 5seed 평균80.70245960, 표본SD .00671195pp, gate3/5통과(seed1/4/5); seed5 80.71406509(+.03150968pp), worstA63−.03889431pp. seed5에서는 두슬롯14weight가 한슬롯15weight보다 +.00669021pp 높음.
- P100 global scan 247/302(81.8%), errors0, controller2700409 생존/CPU5workers. completion 미생성, 기존 모델sharing 대기 SIGSTOP 유지. 원본가중치·채택lite15 변경없음.

## Codex 사용자 lite14 생성 승인 — 2026-10-13
- 사용자 지시로 `lite14_N3_R326plain3456_HRtrain70seed5_20261013` 생성. R8 그대로, E의 seed2 HR392 fold1/4를 train70 seed5 단일 checkpoint(동일25% 그룹)로 교체해 E5, F1 그대로, unique checkpoints14. 후보 valid80.72075530%(+.03819989pp), worst A63−.05187782pp. 기존lite15·모든원본 보존, 새로운 package/manifest/output만 생성.
- ANA 실제GPU0 idle 확인. selected manifest14개 및SHA 검증, portable infer_expert에 단일HR그룹 override 적용(구형두fold를 두번 로드하지 않음), R의기존fresh20/10 출력은동일설정·입력SHA 검증후재사용. 새lite14 E/F source valid20/test10 재현 후 독립valid 채점·저장후보20마스크 대조·test mask 변화 비교. valid 선택개발점수임을 명시, test GT 읽지 않음. P100전역scan은lite15기준 그대로 보존.
- lite14 전체 패키지 생성·검증 완료: 14개 실제 checkpoint(원본 symlink 없음), R8/E5/F1, 2,033,791,825bytes. lite15 대비 308,934,719bytes 감소. manifest checkpoint/metadata SHA 및 code SHA 검증 통과, 원본 미변경.
- 새 portable infer_expert의 E/F raw-image valid20 출력은 저장된 seed5 pair 후보와 20/20장 pixel exact(차이0), 독립 valid80.7207553045%, +.0381998870pp, worstA63−.05187782pp,14/6개선/악화. R fresh20/10은 설정·입력SHA일치 검증 후 재사용. test10 source inference 완료, lite15 대비26,642/9,074,964px(0.293576922%) 변경, test GT미사용/정확도미주장.
- 초기 F 단계에서 빠진 refiner Python dependencies/config를 보완; 실패 로그/JSON 보존, 완료E20은 재실행 없이 CPU F만 복구. code ZIP은 weights 제외, 약952KB. 로컬경로 `experiments/model349_compression/lite14_N3_R326plain3456_HRtrain70seed5_20261013/`, ANA동일명패키지에는14실제가중치 포함. code ZIP CRC·sourceSHA 및 수집한 valid20/test10 마스크30SHA 통과. 기존lite15 유지 보존; 사용자 요청lite14는 별도working모델로 등록, valid로seed선택했다는 제한README기재. P100scan은사용자지정lite15기준 그대로 진행/모델sharing은대기상태유지.

## Codex P100 lite15 전역탐색 완료 — 2026-10-13
- 완료시간 UTC12:48/KST21:48. 302개 후보×23규칙=6,946 case, errors0, controller2700409 종료/새GPU사용없음, release_replaced false. ranking4MB·302개 result/per-image·coverage·baseline_preflight·completion·HR_removal 로컬 수집, archive302JSON 및 각23규칙 대조 통과. 수집경로 `codex_lite15_allmodels_global_20261013/collected/`.
- 기존mean/local gate264 case통과(고유모델264개라는 뜻아님). gate통과 E 추가 최고는 HR336 control cache422 add_E_0.1:80.72214824%(+.03959283pp), worstA59−.02846269pp; E s2f1 교체 cache422:80.72190300%(+.03934758pp), worstA63−.09091226pp. 모두lite15기준의개발screen이며 현재14기준점수가아님.
- R329group→one 형식으로 gate통과한 cache233/593는 baseline에 거의같은 평균(+.00001499pp). archive 역할명이나weight_cost필드만으로 실제checkpoint가 줄었다고 주장하지 않으며 원 producer/평균 여부 점검 필요. R269→HR347 cache262도 +.01537015pp/worst−.08872372pp로 통과하나 기존R347공유인지·TTA/확률provenance확인 전 실제감소판정/채택없음.
- 현재사용자요청lite14는보존. 완료된scan은lite15로 고정됐으므로 유력후보를lite14에적용하려면 새baseline 재검증 필요. 기존 model-sharing 대기프로세스2692046은oldbaseline용이라SIGSTOP유지, 중복후속CPU작업을자동으로시작하지않음.

## Codex lite14 상위 후보 집중 재검사 — 실행 전 등록 2026-10-13
- 사용자 지시: 완료lite15전역탐색 상위후보만lite14기준재검사. ID `codex_lite14_top_candidates_20261013`. 고정shortlist15개 저장예측/34case: 추가top6(각5/10/20%), 대체top6, Fveto top3, R329group분포교체top2(기존국소gate미통과여도구조변경가능성 대조). 중복후보는합쳐15개. 원래E s2f1 자리는현재train70 seed5단독그룹이므로 전체25%그룹교체·halfblend·남은두HR공유로명확히변환. 현R347확률을R269자리에공유하는13weights직접대조1개포함.
- P100실제globalcontroller종료확인, 이전modelsharing2692046 SIGSTOP유지. RAMavailable74GB. CPU전용4workers×2torchthreads, GPU/학습/원본추론미실행. ANA의lite14fresh E20·최종mask20 및seed5원본prob20 전송후현재14 BASE80.7207553045 exact20선행, 입력SHA/채널/후보prob SHA대조. 새규칙의모든예측에서F재계산, mean/worstimage/class·실제가중치추적. oldweight_cost 추정필드만으로축소주장하지않음.
- 결합계수새grid확대없음/기존−.1ppgate유지/valid개발선택의편향명시/후보자동채택없음/기존14·15·원본가중치·Claude기록보존. 실제weight provenance와TENT ensemble여부도확인.
- 집중검사 완료: 15후보/34case, errors0, CPU4workers 종료. parent+workers 모두lite14 E확률 재구성maxabs0.0, 최종mask20 exact, inputSHA20 exact 및candidate probability SHA를이전global과대조통과. mean≥baseline/worstimage≥−.1pp gate17case통과. 15개result·34case perimage/예측마스크·rank·provenance 로컬수집/압축archiveJSON수검증.
- 최고 HR336 CE/Dice control4epoch checkpoint(d91a9fa...) E10%추가:80.74704856%(lite14+.02629326pp), worstA181−.06572767pp,14/6개선/악화. E5%추가:80.74012395%(+.01936865pp), worstA72−.02160410pp,15/5개선/악화. 별도checkpoint1개확인, 기존14에없어서둘다15weights. contextHR336 E10%는80.74663782(+.02588252pp), worstA72−.066130pp로통과. lovaszHR336 E10%80.74576285(+.02500754pp)통과. E20%는주요HR후보모두국소손실−.2pp넘어미통과.
- 14weights유지가능 대체: R322fold3→기존R326 fold3v1,80.72833174(+.00757644pp), worstA74−.032046pp. probe producer는1fold의1view로확인, sourceckSHA427e2c... 기존R326plain3456fold2와다른checkpoint1개가R3221개를대체. v2/v4도+.00749138/+.00673416pp통과. 새로운seed/weight학습없음.
- HR70 전체교체(control/lovasz) −.040299/−.042280pp; 남은두HR 공유하여13weights는−.136888/−.137768pp로미통과. 기존활성R347을R269에도직접공유13weights는80.71312667(−.00762864pp), worstA71−.14932408pp로미통과. 별도newmem HR347 v1으로R269교체는80.73648747(+.01573217pp), worstA71−.076555pp통과이나현재활성R347캐시와다른예측이므로13weights공유재현/producer상세확인전축소주장없음.
- R329 TENT group분포교체상위2는 +.01888754/+.01825549pp이지만worst−.278196/−.231873pp로미통과, 기존audit에서probe_tentensemble count자동단일추정이부적절함을재확인. FAlveto최고+.00608040pp/worst−.017760pp이나추가checkpoint/TTA공유여부확인전모델수확정없음. 현재lite14패키지유지, 선별결과로자동적용없음;추가후보freshsource/test검증은미실행.

## Codex lite13 5seed 및 B4 축약 — 실행 전 등록 2026-10-14
- 사용자 GPU 시드탐색 실행 지시. ID `codex_lite13_E_seed5_reduction_20261014`, ANA GPU0. 실제 GPU0 0%/15MiB 및 학습 프로세스 없음 확인. 다른 담당 search13.py CPU 3worker 보존. 서버 lite13_R322share326_20261014의 fresh valid80.72098350 baseline을 고정하고 exact20 확인 후 평가; 활성 릴리스 자동교체 없음.
- E B4ws fold1/3: 기존 완료 ws4/5/6 재사용+신규 ws7/8 각2fold 학습(총5시드). 원래fold부모 유지, BNfixed2000step/원래LR3e-5/마지막checkpoint/학습중valid미사용. E HR392 seed1fold4: 완료5001/6001 재사용+새7001/8001/9001 scratch70epoch 원래분할/레시피(총5시드). 다른완료학습 중복하지 않음.
- 각5시드 current lite13 E/F에서 결합계수 고정 채점/mean SD range/perimage 기록. R326 기존4sets×fold2 일반가중치 확률을현재 공유두역할에 재적용(새학습없음). R329 팀원시드탐색중간기록이 있으므로 원본처음부터학습을중복하지 않고 최종 결과 확인후현lite13재평가 대상으로유지.
- 축약검사: E B4ws2→1(현재부모와5seed 각fold) 및 R329 f1/3/4의7부분집합, joint조합; 원래입력/채널/TTA provenance 검증. fresh 재현 전 저장확률screen으로 명시. train70 warm/EMA/SWA/soup 기각 결과는 보존해동일레시피반복없음. test GT 미사용/valid 개발점수/국소gate mean비하락 및worst>=−.1pp 유지.
- 01:52 KST 실행확인: controller1309789 생존, baseline fresh lite13 80.7209835046/20마스크exact 통과. hr_seed7001 새학습이GPU0 사용률90%/6533MiB에서 실제진행,53images/159steps epoch/targetwidth1278 원래fold4조건확인. 총7순차GPU작업 및각valid추론→고정lite13 CPU5seed편차/축약검사 연결. 기존3CPU search13와대기프로세스 untouched.

## Codex lite13 재조정 test10 비교 — 실행 전 등록 2026-10-14
- 사용자 요청 원래lite13 vs retune_full(기존13가중치/계수만변경) test10 예측 비교. ID `codex_lite13_retune_test10_20261014`. ANA CPU전용/현재HR7001 GPU학습 유지, 저장된동일test raw component probabilities 재사용. full valid선택계수 동결/test보고후추가튜닝없음/GT미사용/정확도우열주장없음.
- 원래lite13 10마스크pixel exact 및inputSHA를선행 확인후새R/E/F 재결합, 변경률·클래스전이·경계거리·변경영역을측정하고 raw/기존/재조정/변경 시각자료수집. 기존모델·가중치·릴리스·공유기록보존.
- lite13再조정test10 CPU비교완료: 기준fresh10마스크pixel exact/inputSHAunchanged/GT미사용. total38,776/9,074,964px=0.42728544%변경,99.71116%가원래경계2px이내. A51 7428px(.63940666%)최대,A1245749(.53416957%),A1275718(.38777669%). largest변경연결영역전체최대101px(A51). 原E분해재구성maxabs0 exact; 선택retune_full과동일per-foldB4FP16평균캐시사용(원래fresh가족과maxabs.00048828),저장확률재결합screen으로명시/새GPU추론없음. 原/新labels각10/그림20/비교JSON로컬수집. 시각검토A51/A124의변화는주로경계·미세조직,정답없이우열미확정/활성모델미교체/GPUseed학습지속.

## Codex lite13 재조정 비교 이미지 ZIP — 실행 전 등록 2026-10-14
- 사용자 요청 valid20 이미지별 mIoU수치증감·test10 예측차이를눈에보이게 시각화하여ZIP제작. ID `codex_lite13_retune_comparison_zip_20261014`. 현재GPU시드학습유지/CPU저장확률만재결합/retune_full계수동결/새계수탐색없음. valid20 기준마스크exact및기존retune full델타재현 확인; GT있는valid만mIoU평가/test는변경량만. 입력·GT·원/재조정마스크·강조변경지도·확대컷·수치표를수집/이미지QA후ZIPCRC검증. 원본릴리스미변경.
- 사용자후속질문: E HRNet추가+확률재조정동시valid점수 확인. 기존 retune_full 선택계수 동결후 HR336 seed3410/control/context 3후보×추가5/10% 6case CPU비교. 새계수최적화아님/14weights/기존13보존/test추가사용없음. ZIP은앞서요청한원래13vs재조정13만포함해범위혼동방지.
- 사용자비교ZIP제작완료: Model349_lite13_vs_retuned_valid20_test10_20261014.zip,66,916,914bytes/228members. valid full20+zoom40/testfull10+zoom20 및요약PNG2(총92시각자료),raw/labels/validGT130PNG·수치TXT/JSON·README/체크섬포함. valid基準20exact/재조정delta+.0583570631 exact,12개선8악화. 모든PNG verify/전송assetSHA/ZIPCRC/압축내모든memberSHA검증 통과. A153 valid손실확대·A51test확대·20행valid요약시각QA확인. ZIP범위는原lite13 vs retune13이며HR추가14후보는포함하지 않음.
- HR추가+既存retune full동결6case完: 최고seed3410 E10%80.78743517%(原13+.06645166pp/retune13+.00809460pp),worstA153−.08091698pp/14weights. control5%80.78727379/worst−.07118372,context5%80.78591375/worst−.06592280 gate통과. control10%80.78428410/worst−.15815930·context10%80.78139691/worst−.12174857 gate미통과. seed3410 5%80.78514364/worst−.06897907. 新계수재최적화없음/追加14候補test未실행/원본모델미변경. combo_results로컬수집.

## Codex lite13 모든 구성원 시드편차 재집계 — 2026-10-14 실행 전 등록
- 사용자 전체표요청. ID `codex_lite13_all_seed_statistics_20261014`, ANA CPU저장예측재결합/기존GPU학습보존. 완료N3/326/H70 통계수집 + Res269 5seed 저장확률·F5개기존checkpoint CPU재평가·B4ws현재2fold완료3seed·HR392현재단일f4완료2seed 재평가. 이전구성의HR347/restart61/R329 수치는현lite13 측정치로혼용하지 않고누락범위명시. fulltable에모든현구성원표시/입력원본보존/추가학습중복없음.
- 全편차재집계완료: 현재lite13에서7계열재평가(Res2695/F5/B4ws완료3/HRfold4완료2/N3 4/R326공유4/H70 5). 原20maskexact80.72098350확인,새학습/GPU추론미실행. 269부모차이명시/HR347·restart61·R329의현lite13동일조건결과는부재하여과거수치로대체표기하지 않음. 全mean/sampleSD/range/perimageJSON로컬수집/진행중GPU5seed큐보존.

## Codex lite13 B4 신규 축약 방법 — 실행 전 등록 2026-10-14
- 사용자 이전시도 제외하고E B4ws/R B4width4000 축약재탐색 지시. ID `codex_lite13_B4_new_reduction_20261014`. P100 CPU전용,ANA 현재5seed학습유지. 제외:단순fold부분집합자체재탐색/전체train70 단일대체/동일warmstart·EMA·SWA/전체weight soup·BNrecal/기존모델전체그룹대체/TTA·다중scale 제거변형/증류.
- 新후보: 기존남길subset의확률에①온도T=1.15/1.30/1.50(과신완화)②원본밝기경계를보존하는3x3확률bilateral혼합λ=.15/.30(σgray=.05). R3293→2/1의6subset×5보정=30 및B4ws2→1의2subset×5=10,총40개신규screen. 결합계수R/E/F동결,단순subset8개는새탐색후보가아닌matched control만. valid개발후보선택으로명시/독립성능주장없음/사전mean비하락·worst>=−.1pp 유지.
- 다른新방향:共享B4encoder+fold別decoder. 원래5모델의encoder/decoder state keys·byte비율·architecture/preprocessing호환성을CPU확인. 전체weight soup와구분하며현재서로다른encoder에학습된decoder를그대로공유해성능유지한다고가정하지 않음. 실제head수/encoder数/storage별보고/단일파일직렬화를1모델축약으로주장하지 않음.
- lite13 fresh20labels 및cfg·E exact20 baseline을P100전송검증후시작/예측별채널MAIN변환중복없음/GPU사용·타사용자중지없음/원본가중치·릴리스보존.
- 新screen R32936比較(30新候補+6control)完/通過0。B4ws段階でfork後Torch CPU refiner進入worker2800770/71がfutex待機・CPUtick不変を確認。自分のcontroller2800304とその2childだけ停止し、36result再利用で残りE12比較を同一新プロセス内逐次実行へ修正。元コード/ログ/進捗保存。GPUシード学習・他ユーザーprocessは触らない。
- 新screen완료:新40/control8/errors0,baseline13 fresh20exact. temp·밝기edge보정 통과0. B4ws f1 T1.5 총12weights80.74182722(+.02084372pp)/12개선8악화이지만worstA181−.24194186pp(최악class−.98765476pp) 기준미통과. R329 f1·3 edgeλ.15 총12weights80.66788999(−.05309352pp)/worstA186−1.19978354pp. R329국소손실가장작은f1·4 T1.30도80.62273731/worstA74−.54358034pp 미통과.
- P100 deadlock복구후48result모두정상수집/40ranking·8control·48개caseJSON수대조/원본13유지. 同參數shape공유encoder메모리점검은R329244.24→102.85MB(−141.39MB/57.89%),B4ws162.83→92.13MB(−70.70MB/43.42%) 이론tensor bytes. 실제encoder值다름/decoder정렬학습과원본추론검증미실행/출력3·2분기유지/실제모델수축약·성능유지주장없음/OOF독립성변화유의. 현GPU seed8001 학습불변.

## Codex lite13 B4 인코더 공유 실제 추론 — 실행 전 등록 2026-10-14
- 사용자 인코더공유테스트 지시. ID `codex_lite13_shared_encoder_20261014`, ANA GPU0. 실제GPU0 seed8001학습96%/6533MiB 확인, 기존5seed controller1309789·큐보존. 새실험은현재GPU큐completion후유휴확인에서만실행/다른사용자작업종료없음.
- phase1 재학습없는3공유범위+합성대조: R329(f1encoder+f1/3/4decoder),E B4ws(f1encoder+f1/3decoder),각그룹공유동시적용,전체5head가R329f1encoder공유. fold1 anchor사전고정/계수·F고정/validepoch선택없음/증류없음. Ridentity/E D4 및width4000/ZCDF 동일실제입력처리검증. 전체공유는identity특징을R/E두경로에서재사용하고E나머지7view만추가encoderforward.
- 전체13현재실물ckSHA 검증/heads 및encoder shape·동일입력전처리·SMP decoder호출규약 확인. customencoder→decoder→seghead 단일head출력과원래모델출력수치대조후원본valid20新추론. original group20cache와固定lite13마스크exact를선행검증/공유패키지encoder+decoder분리저장·실제bytes 및고유encoder/head数표기. decoder3·2개유지이므로13→N개 full-model삭제로주장하지 않음.
- 이미지별·클래스별valid 손실·encoder呼出数·시간/VRAM 측정. 성공후test10 비교,실패면현재13유지. phase1결과를보고decoder정렬학습필요성판단; 현시점정렬학습미실행. sharedanchor는다른fold held이미지를본가능성이있어OOF독립성유지주장없음/valid는반복개발점수/testGT미사용.
- 인코더공유구현·사전검증완료: 현재13 ck/metadata SHA검증,5모델입력metadata완전동일(width4000/ZCDF),R/E make_input同一,CPU64x64 encoder→decoder 분리forward 原모델差0。baseline20maskexact80.72098350及E B4family分解差0 확인. 물리공유package作成R_shared103,118,988bytes(1encoder3head),E_shared92,381,987(1encoder2head),both_separate195,500,975(2encoder5head),all_shared124,592,990(1encoder5head)。新controller1687253 GPU기존큐completion대기/학습재실행없음. valid통과시test10原추론+원래13exact 비교 자동waiter1708989 등록,미통과면test추론skipreceipt。성능결과는아직없음/decoder보정학습미시작/기존릴리스·GPU학습보존.

## Codex 사용자 인코더공유 valid20 최우선 변경 — 2026-10-14
- 사용자 기존seed큐보다인코더공유valid20 먼저지시. 현재seed9001 HRtrain 학습/epoch별last.pt+train_state.pt(model/optimizer/scheduler/scaler/RNG)저장지원확인. GPU메모리해제를위해완료epoch 체크포인트안정저장→우리seedcontroller/child만보류·종료→sharedencoder priority override허용. SIGSTOP만으로는GPU6.5GB가남아있으므로단순중지상태에서공유추론동시실행하지 않음.
- 완료7001/8001학습·추론재사용. 未완료9001은기존동일outdir에서--resume으로다음저장epoch부터이어감; 이후ws7/8원래큐순서유지. 중단checkpoint/로그·상태별도백업/원본학습파일·릴리스유지. 인코더공유valid완료및조건부test후우리resume supervisor연결,다른사용자작업은미중단.
- 우선순위변경실행완료: HR9001 완료34epoch last.pt/train_state.pt model exact 확인후4파일(optimizer/scheduler/scaler/RNG 포함state925MB)별도priority_pause_epoch34 백업. 검증된우리controller1309789/child1713563 및그직계DataLoader들만종료해GPU메모리해제/other untouched. 新sharedcontroller1809211 실제GPU100%/7.2GB 추론(R_shared20完/E_shared進行),priority marker는user지시기록이며seed완료를위조하지 않음. 新resume supervisor1809212는sharedvalid/조건부test후9001 --resume35epoch부터기존outdir재개,완료7001/8001·prob캐시재사용/이후B4ws原queue유지. GPU단일head分리controlも差0 확인.
- 공유encoder우선valid20 평가완료/4구성각20마스크독립GT채점/原baseline80.72098350 exact. R내부공유80.53295506(−.18802844pp),worstA186−1.99908508;E내부공유80.69000917(−.03097434pp),worstA181−.16450837;각그룹동시80.50167812(−.21930538pp),worstA186−1.99514892;전체공유80.46341007(−.25757343pp),worstA186−1.97198826. 모두gate미통과/학습없는직접공유를현재13에채택안함/test10후속은skipreceipt保存.
- source/data SHA불변/CPU·GPU分리forwardcontrols差0/E分解exact20/全80 PNG·결과·timing·receipt로컬수집검증. raw新B4공유부분20장만의시간R10.85s/E42.09s/전체43.88s이며전체13속도·원래GPU와의정밀비교로주장하지 않음. encoder forward 영상당R1/E8/all8확인. decoder정렬학습미실행. seedresumecontroller1824110が9001從epoch34→35 --resume開始,7001/8001再学習なし/원래ws7/8후속큐유지。

## Codex 사용자 seed 보류·BN별 공유/부분공유 우선 검증 — 실행 전 등록 2026-10-14
- 사용자 seed탐색미루고새방법확인지시. ID `codex_lite13_B4_BN_partialshare_20261014`,ANA GPU0. 현재resume parent1824110/child1826464 HR9001 ~60epoch 실제확인. 우리resume supervisor1809212만종료해자동resume차단; 다음완료checkpoint 안정저장·모델/state exact백업후우리학습만보류. 이번검증후seed자동재개없음(새방법우선).
- 新test①Conv2d weight/bias만anchor fold1로공유하고fold별BN affine/runningstats·decoder원래값유지(R329/E B4ws/각그룹동시/전체5공유4case)②stem+앞4/8/16block만공유하고뒤encoder·decoder는각fold유지(R3/E3/동시3case). 총13case,신규학습·증류·valid계수튜닝없음/原13고정. BN 공유시BN값exact·Conv参数指针실공유확인;부분공유최대layer구조검증.
- BNP별출력이달라encoderforward를fold별수행하므로이번prototype는저장량절감검증이며1회encoderforward/추론가속으로주장안함. 실제unique parameter·buffer·저장bytes 및20GT독립scores/local loss/新raw probabilities/hash 확인. input/TTA는Ridentity E D4/ZCDF width4000로동일. gate mean>=原13/worst>=−.1pp. 통과시test예측비교/原릴리스미변경/조건부valid개발점수.

- BN별/부분공유 valid20 완료확인(2026-10-14 04:20 KST): 13case 모두 평균비하락+최악이미지−.1pp gate미통과. BN_R80.25443937/BN_E80.63016818, 각각−.46654413/−.09081533pp. 부분4block E80.71285080(−.00813270pp),worstA181−.062485pp이나 저장tensor절감149,464bytes뿐이며fold/head수유지. 부분8block E80.69516864(−.02581486pp),worstA191−.098224pp/642,896bytes절감. 국소손실이작아도fold삭제가아님. 조건부test는전부gate실패로skip완료,seed보류유지/릴리스미변경. results.json 로컬수집. 재학습없는직접공유의한계와공유구조정렬재학습을구분함.

## Codex lite13 남은 구성원 시드탐색 — 실행 전 등록 2026-10-14
- 사용자 남은구성원 시드탐색 지시. ID `codex_lite13_remaining_seed5_20261014`, ANA GPU0 현재0%/15MiB/학습없음 확인. restart61·HR347·R329 각각5seed 목표, 현lite13 계수/나머지구성고정 valid20 재평가. 기존시드학습/예측을우선복구·재사용하고 부족분만학습, 팀원R329와동일시드 중복학습금지. 269 배포부모와P100탐색부모가다르므로배포부모확보여부조사후동일조건평가가능범위명시. 현재준비중.
- 원본데이터·가중치·lite13릴리스·Claude기록보존,valid는학습/epoch선택에사용하지않고고정레시피 시드효과평가만. P100다른GPU작업미중단/새GPU작업없음. CPU/GPUbaseline exact 선행,입력SHA·metadata·채널순서 확인.
- 범위구체화: HR347 fixed original336 부모에서4401~4405 각12epoch last, restart61 fixed originalrestart51 부모에서4411~4415 각4epoch original valid-best 절차, R3291757~1761 각각fold1/3/4 16000step scratch(ImageNet 초기화) 총25학습stage. 팀원1557~1562와겹치지않음/GitHub HEAD dc9395e은329 fixedseed0만포함. restart61은원래레시피가valid-best이므로직전등록의“epoch선택미사용”에서예외를명확히수정: 기존선택절차보존/새결합계수·새selection grid 추가없음/valid편향포함조건부편차. 전체부모chain 재학습이아닌현재부모고정 마지막stage 시드효과로명시. 269정확배포model257 부모는현재미확보/경로질문pending,다른계보를동일시드편차로보고하지않음.
- 실행확인: controller2532592, baseline80.72098350/20maskexact 및split trainfold1=52/fold3=53/fold4=53 기존E분할과동일 확인. 준비중타작업b4tta/probe_ws_r2.py PID2528739가GPU를사용해정상종료까지대기(미중단),종료후우리첫HR347_seed4401 학습시작. jobs/scope/preflight/splitproof 로컬수집. 총15seedgroup/25trainingstage 및매group原R GPU추론→fixed lite13 CPU평가/이미지별mask저장자동연결. 완료269부모확보없이는전체구성원동일조건탐색완료로주장안함.
- 2026-10-14 12:40 KST 실제확인: HR3475시드 학습/valid/fixedlite13채점 완료, mean80.65535079 SD.04917475 range.12964879pp/전부原13보다낮음. seeds4401~4405 final80.70902222/80.68444629/80.64330021/80.57937343/80.66061181,최악이미지모두A42(−.33296/−1.10024/−.93996/−1.02402/−1.34857pp). 現347이336자리도공유하므로과거347단독기여SD와다름.
- controller2532592는restart61seed4411學習4epoch완료후valid3/20에서다른GPU프로세스2923834(11.49GiB)와경합해OOM실패/정상완료아님/R329未시작. 해당타프로세스미중단. 原sourceSHA불변검증,완료HR5 및restart61학습receipt/3개pickle예측재사용으로修正:base추론영상마다GPU다른PID유무대기/이미지별checkpoint리셋특성으로누락17장만계속. oldfailure/controller/code/inputSHA보존 및recoveryreceipt作成;新controller805404 재개. GPU現유휴,원래release미교체/이미완료학습재실행없음.
- 2026-10-14 사용자R329재학습불필요지시: R329未시작상태확인후cancel_R329.json등록/새큐에서15개B4stage제외. 현재restart61seed4414 child861564는중지않고完了보존,검증된우리parent805404만SIGSTOP하여큐진입차단. finish_current_then_resume.py supervisor895631가child正常exit 확인/receipt보존→우리oldparent종료→R329제외한controller재개하며restart61seed4415및평가는이어감. 원본release·다른사용자process미변경.
- 팀원GitHub再확인: giveItToHyeonSeok main HEAD dc9395ec52c9b47bb6886a982d4e6dd0e366a503(10/13 11:04:32KST)로이전과동일,branch main외없음/tags·releases없음. metadata/additional_models.json는R329seed0 folds1/3/4고정/새시드비교점수null/no_seed_search_for_this_component=true. 사용자제공1557~1562탐색자료가완료됐다는정보는존중하되현재저장소에미게시로분리. 새R329가중치다운로드/학습없음.
- 2026-10-14 13:02 실제완료확인: controller896074 정상종료completion=true/inputSHA불변/HR347·restart61각5seed/取消R329실행없음/269原parentpending/릴리스미교체. restart61 fixedlite13 finalseed4411~4415=80.72065083/80.72511564/80.72341391/80.72315333/80.71936201,平均80.72233914 SD.00230438 range.00575363pp. worstA71−.06095/−.13856/−.11997/−.07220及A74−.07088pp. 本stage 부모고정 및legacyvalid-best선택조건의편차이며전체chain변동/독립일반화추정아님. 結果+completion+transition로컬수집,현재GPU PID948512는우리종료작업과別사용자작업/미중단.

## Codex 팀원 R329 6시드 lite13 재평가 — 실행 전 등록 2026-10-14
- 사용자제공 /Users/hyunseok/Downloads/R329_seed_search.zip 확인:seed1557~1562/fold1·3·4 가중치18개 및lite19a기준valid/test변화기록. ID `codex_lite13_team_R329_seed6_20261014`. 재학습없음/ANA GPU0현재compute없음확인후valid20 B4group만6회추론→lite13 고정R/E/F 재결합/독립GT채점. 팀원summary는lite19a이므로현재13값으로혼용금지. original13baseline20maskexact/sourceSHA/18ckZIP SHA/args fullrecipe/train16000完了확인/원본릴리스보존.
- archive18 checkpoint CRC/SHA 및각train.log16000step/args검증통과, ZIP SHA515ffcecee37491a49b84df1883fc461e06ce99e35a49c2b6d63570ab4fa98a1. 팀원lite19a6seed平均80.37853223/표본SD.09715439/range.25595848pp,baseline80.52364212/전부低下. currentlite13수치아님. ANAcontroller1098461 실제생존/1.47GBarchive전송대기phase 확인;전송완료SHA→안전extract→baselineexact→R3296group原추론(identity TTA)→고정13재결합 자동연결/새학습없음.
- 2026-10-14 13:38 실제확인: teamR3296seed lite13再평가正常完了/새training0/18ckSHAunchanged/release未교체/testGT未사용。baseline20exact80.72098350,seed1557..1562 final80.47466325/80.60141226/80.38263971/80.58943974/80.46068210/80.64442258,平均80.52554327 SD(sample).10118135 range.26178286pp. 原13보다모두低下(−.24632/−.11957/−.33834/−.13154/−.26030/−.07656pp),최악A42−1.42803/A186−1.16449/A71−3.16684/A186−1.66025/A71−1.84110/A42−1.20879. 종전팀원lite19a값과분리/새계수튜닝없음/최고시드교체미채택. 結果/完了/baselinepreflight 로컬수집,ANA현재compute없음。
