# 보관 범위와 출처

2026-10-22 수집 시점의 스냅샷입니다. 사용자의 요청에 따라 개인 저장소는 공개로 유지하고, **모든 학습 가중치는 제외**했습니다. 기존 팀 저장소와 서버 원본은 수정·삭제하지 않았습니다.

## 저장한 자료

- 서버 실험 45개 폴더의 코드·설정·README·학습 로그·평가 CSV/JSON과 실행 소스 스냅샷.
- 서버 결과 문서 47개(기존 색인 포함), 팀 GitHub에만 남아 있던 과거 결과와 실험 자료.
- 최종 제출 lite13의 학습·추론 코드, 개발 환경, 모델링 결과서와 예측 마스크.
- Fusion805 보고서의 DOC/DOCX/PDF/MD, 그림·표와 생성 코드.
- 비교 그림, 데이터와 큰 문서는 Releases에 저장하고 SHA-256을 확인한 뒤 원래 상대 경로로 복원.
- 이전 GitHub 업로드 작업의 최상위 패키징 코드와 검증 기록도 `server/artifacts/`에 보관. 그 아래 임시 clone·도구 설치물·중복 가중치 ZIP은 제외.
- 작은 파일은 Git에 보관; 동일한 대용량 파일은 SHA-256 기준으로 중복을 제거해 한 번 저장하고 모든 경로의 대응을 기록.

## 제외한 자료

- `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx` 등 모델 파일, 직렬화 체크포인트와 모델 가중치가 포함된 ZIP.
- 기존 Git LFS 가중치 포인터와 가중치 배포용 분할 압축 파일.
- 실험 중간 확률·특징 배열, 반복 복사된 실험 이미지 캐시와 Python/TensorBoard 캐시. 데이터 루트의 전처리 배열은 자료로 보관.
- 보고서 전용 도구 설치물, 가상환경, `.git` 내부 파일, 로컬 인증 정보.
- 팀 저장소의 이전 커밋 전체와 가중치 전용 Releases는 복제하지 않음. 현재 스냅샷의 출처 커밋은 아래에 고정.

서버의 중간·최종 체크포인트는 서버에 그대로 남아 있습니다. 이 저장소는 서버 디스크 전체의 완전 백업이 아니며 가중치를 포함한 원래 실행 상태를 단독으로 복원할 수 없습니다.

## 팀 저장소 스냅샷

| 원본 | 커밋 | 복사본 |
|---|---|---|
| [CNUKoMaP/HRNet_best](https://github.com/CNUKoMaP/HRNet_best) | `48fe7ac2aee966c2d7c862402b4a2d3648b5450d` | [자료](../team/HRNet_best/) |
| [CNUKoMaP/bsh](https://github.com/CNUKoMaP/bsh) | 빈 저장소 | 파일 없음 |
| [CNUKoMaP/giveItToHyeonSeok](https://github.com/CNUKoMaP/giveItToHyeonSeok) | `12b380240c6eac924dfdcdc6e05300584c505946` | [자료](../team/giveItToHyeonSeok/) |
| [CNUKoMaP/lhs](https://github.com/CNUKoMaP/lhs) | `b1e978bbd02019f6759b4abf689dd766bcb9838e` | [자료](../team/lhs/) |
| [CNUKoMaP/model349-lite19a-r70-swa](https://github.com/CNUKoMaP/model349-lite19a-r70-swa) | `802ef91677155492eb3e1d1b3758745194e36c31` | [자료](../team/model349-lite19a-r70-swa/) |
| [CNUKoMaP/model349](https://github.com/CNUKoMaP/model349) | `31428158c96d5f137f8d5c0cc3d747495ad02bca` | [자료](../team/model349/) |
| [CNUKoMaP/sppark](https://github.com/CNUKoMaP/sppark) | `e0756446f18959e4eaf28b860d56f3d8604ce441` | [자료](../team/sppark/) |

SWA 코드는 해당 팀 저장소의 `v1` Release에서 가져왔고, 최종 제출 코드 ZIP은 lhs의 `submission-final-20261015` Release에서 가져왔습니다. 가중치 Release 첨부 파일은 공개하지 않았습니다.

## 검증 목록

| 목록 | 용도 |
|---|---|
| [files.json](../manifests/files.json) | 복사한 원문 파일의 경로·출처·크기·SHA-256 |
| [assets.json](../manifests/assets.json) | Release 파일, 해시와 원래 경로 대응 |
| [omitted_files.json](../manifests/omitted_files.json) | 조사 대상 중 제외한 파일과 사유 |
| [source_repositories.json](../manifests/source_repositories.json) | 팀 원본 커밋·브랜치·태그 |
| [server_symlinks.json](../manifests/server_symlinks.json) | 원래 서버의 심볼릭 링크 기록 |

새로 작성한 README·색인·복원 도구는 보관용 문서이며 원본 파일 해시 목록에는 들어 있지 않습니다. 원본 `.gitignore`, `.gitattributes` 등은 실행되는 설정을 전파하지 않도록 `.source.txt`를 붙여 내용만 보존했습니다.

원본 문서의 서버 절대 경로, 팀 저장소 링크, 당시 지표 정의와 오탈자는 수정하지 않았습니다. 과거 문서에 남은 가중치 링크는 이 개인 저장소에 가중치가 있다는 뜻이 아닙니다.

## 저작자와 외부 코드

자료의 기여 주체는 원본 CNUKoMaP 팀 저장소와 서버 실험에 남아 있습니다. 팀 자료를 개인 단독 저작물로 표시하지 않습니다. 외부 라이브러리·모델·글꼴의 기존 라이선스와 출처 파일은 함께 보존했습니다. 이 아카이브에서 팀 전체 코드나 대회 데이터에 새 라이선스를 일괄 부여하지 않았습니다.
