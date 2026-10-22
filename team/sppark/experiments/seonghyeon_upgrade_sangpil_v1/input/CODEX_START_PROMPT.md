# 서버 Codex에 전달할 작업 지시문

이 파일과 `alphase_b4_codex_spec.md`를 기존 프로젝트의 일반 문서로 둔다. 기존 `AGENTS.md`나 설정을 덮어쓰지 않는다. 아래 지시문을 Codex에 전달한다. Codex가 현재 저장소 안에서 명세서 파일의 실제 위치를 먼저 찾도록 지시한다.

```text
현재 저장소의 AGENTS.md, README와 기존 EfficientNet-B4 U-Net 학습/추론 코드를 먼저 읽고,
현재 저장소에서 alphase_b4_codex_spec.md의 위치를 찾아 읽고,
그 명세에 따라 개선 프로젝트를 실제 코드로 구현해줘.

목표는 작은 미세조직과 복잡한 혼합조직, 특히 Al3Ni → EutecticSi 오류를 줄여
원본 이미지별 평균 validation mIoU를 높이는 것이야.
대형 backbone으로 교체하거나 공정 Si 가중치를 무조건 증가시키는 일부터 시작하지 마.

먼저 기존 저장소/환경/데이터/체크포인트를 확인해서 repo_audit.md,
resolved_legacy.yaml, dataset/checkpoint manifest를 작성해.
기존 코드·100개 seed checkpoint·완료된 결과는 보존하고 새 실험을 별도 경로에 기록해.

다음 기능을 명세서 P0–P7 순서로 구현하고 테스트해:
- 기존 seed32 추론/지표 재현과 100-seed 확률 앙상블
- native ROI 기하와 geometry-only 대조군
- 상대 밝기 TextureStem, 넓은 ContextCNN, zero-init 잔차 분할 head
- train-only mixed/hard sampler와 gradient가 연결된 native loss
- best/last/EMA/resume 및 paired-seed 실험
- strict OOF teacher cache, lineage validator와 4클래스 native refiner

모든 branch와 loss는 독립적으로 켜고 끌 수 있게 해.
E03b detail-only 잔차 대조군을 만들어 texture/context의 추가 정보 효과를 분리해.

필수 규칙:
1. 클래스 순서는 Al / Al3Ni / PrimarySi / EutecticSi다.
2. 지표는 원본 이미지마다 4클래스 IoU를 평균한 뒤 이미지 평균이다.
   union=0이면 1이고, GT에 없지만 예측하면 IoU=0이다. Global mIoU와 구분해.
3. 4000px 확대→1536 crop과 native ROI는 정확히 같은 처리라고 가정하지 마.
   D4, 상대 밝기 sigma, context 좌표, native 복원에 단위/좌표 테스트를 만들어.
4. 같은 train70을 본 seed 모델 100개는 OOF가 아니다.
   all-train checkpoint에서 warm-start한 fold teacher도 strict OOF가 아니야.
5. valid20/A71 정답을 train sampler, 입력 정규화, refiner 학습에 사용하지 마.
6. 기존 모델과 새 코드의 동작 차이를 숨기거나 점수를 맞추는 보정을 넣지 마.

모르는 라이브러리/API/경로/학습 세부값은 원본 코드부터 확인해.
탐색해도 알 수 없는 필수값만 질문하고, 차단된 통합 테스트는 BLOCKED로 남겨.
단위 테스트와 smoke test까지 실제 실행해. 가짜 데이터로 shape test를 통과한 것을
실제 B4 체크포인트 재현 또는 성능 향상이라고 보고하지 마.

긴 전체 학습/100개 추론 sweep은 먼저 실행 계획을 만들고 --execute 옵션으로 분리해.
이미 허용된 실행은 기존 서버 자원 규칙 안에서 수행하고, 다른 작업을 종료하지 마.
각 단계마다 progress.md에 변경 파일, 명령, 테스트, 실험 결과/미실행 항목을 기록해.
분석 계획만 내놓고 끝내지 말고 가능한 단계의 실제 구현을 진행해.
```
