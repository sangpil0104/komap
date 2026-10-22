# Model349 lite19a_r70 + SWA 실험 패키지

R의 B4 fold 6개를 train70 모델 3개로 바꾸면 valid20이 −0.54 떨어집니다. 대부분은 "모델 1개 vs 여러 개 평균" 차이입니다.
이 패키지는 **SWA**(같은 학습 경로의 가중치 평균)로 모델 수를 늘리지 않고 그 효과를 되찾을 수 있는지 확인합니다.
배경, 측정값, 판단 기준은 코드 zip 안의 `r70/swa/README.md`에 있습니다.

## 받을 파일

모두 **Releases → v1**에 있습니다. 가중치 zip 2개는 GitHub 업로드 제한 때문에 400 MB 조각(`.parta`~`.partd`)으로 나눠 올렸습니다. 받은 뒤 `cat`으로 합칩니다(아래 실행 참고).

| 파일 | 내용 | 크기 |
|---|---|---:|
| `Model349_lite19a_r70_swa_code.zip` | 코드 (기존 r70 번들 + `r70/swa/train_swa.py`, `r70/run_swa.sh`) | 2.5 MB |
| `lite19a_weights_R.zip.part{a,b,c,d}` | lite19a 공개 R 가중치 10개 (+ args.json) | 1.3 GB |
| `lite19a_weights_assets.zip.part{a,b,c,d}` | lite19a 공개 E/F 가중치 9개 (+ args.json) | 1.6 GB |
| `Model349_lite20_r70_B4_{322,326,329}_weights.zip` | train70 B4 3개 (seed 0, 21,000회; CNUKoMaP/sppark에서 복사) | 72 MB × 3 |
| `SHA256SUMS.txt` | 체크섬 (합친 zip 기준) | |

데이터(train/valid/test)는 포함하지 않았습니다. 대회 데이터를 그대로 쓰면 됩니다.

## 실행

```sh
mkdir -p ~/swa && cd ~/swa
gh release download v1 -R CNUKoMaP/model349-lite19a-r70-swa      # 또는 웹에서 Releases → v1 파일 전부 받기
cat lite19a_weights_R.zip.part? > lite19a_weights_R.zip && cat lite19a_weights_assets.zip.part? > lite19a_weights_assets.zip
sha256sum -c SHA256SUMS.txt                                          # 7개 모두 OK여야 합니다
unzip -q Model349_lite19a_r70_swa_code.zip                         # -> Model349_lite19a_r70_swa_bundle/
mkdir -p WEIGHTS && unzip -q lite19a_weights_R.zip -d WEIGHTS && unzip -q lite19a_weights_assets.zip -d WEIGHTS   # -> WEIGHTS/R, WEIGHTS/assets
mkdir -p PLAIN_B4 && for m in 322 326 329; do unzip -q Model349_lite20_r70_B4_${m}_weights.zip -d PLAIN_B4; done  # -> PLAIN_B4/Model349_lite20_r70_B4_*/

# DATA: train/{images,masks}(70), valid/{images,masks}(20), test/images(10)
cd Model349_lite19a_r70_swa_bundle
bash r70/run_swa.sh /path/DATA ~/swa/WEIGHTS/R ~/swa/WEIGHTS/assets ~/swa/PLAIN_B4 ~/swa/runs 0,1,2
```

- GPU 목록의 GPU마다 B4 하나씩 동시에 학습합니다. 시작할 때 GPU가 비어 있어야 합니다. 1장만 있으면 `0`을 주면 되고, 학습이 3배 걸립니다.
- GPU 메모리는 B4 학습 1개당 약 8 GB 이상이 필요합니다(crop 1536, batch 2).
- 중간에 끊기면 같은 명령으로 다시 돌리면 됩니다. 끝난 단계는 건너뜁니다.
- 환경: 기존 Model349 환경(torch, segmentation-models-pytorch 0.5, timm, albumentations, opencv, pyyaml). 번들 안의 requirements 파일을 참고하세요.
- 시간: P100 3장 기준으로 SWA 학습 약 1.2시간, 추론(SWA/plain × lite19a_r70/lite19a × valid/test) 약 2시간입니다.

## 보내 줄 결과
- `~/swa/runs/swa/summary.txt`
- `~/swa/runs/plain/summary.txt`
- `~/swa/runs/swa/b4_full70/*/train.log`의 마지막 줄 (`SWA done: ...`)

SWA 설정(4,000회, 학습률 1e-4 고정, 500회마다 평균, BN 300배치)은 결과를 보기 전에 정한 값입니다. 바꾸지 말고 그대로 돌려 주세요.
