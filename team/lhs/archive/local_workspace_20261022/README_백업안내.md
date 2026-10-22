# 로컬 작업 폴더 백업 (2026-10-22)

로컬 작업 폴더(`KoMaP/gurnard`)를 지우기 전에 남긴 백업입니다.

- 이 폴더: 5MB 미만 텍스트 파일(코드, 문서, 설정, 결과 JSON·CSV 등). 원래 폴더 구조 그대로입니다.
- 저장소에 같은 내용이 이미 있는 폴더 3개는 중복이라 뺐습니다: `Model364_reproduction_bundle/`, `Model349_lite12_base/`(→ [`archive/bundles/`](../bundles/)), `miou_policy_audit/`(→ [`archive/docs/miou_policy_audit/`](../docs/miou_policy_audit/)).
- 릴리스 [`local-workspace-archive-20261022`](https://github.com/CNUKoMaP/lhs/releases/tag/local-workspace-archive-20261022): 나머지 6,634개 파일(PNG, 로그, npz·npy 확률, pt 가중치, docx·pdf 보고서 초안, zip·tar.gz 묶음 등, 약 2.65GB). tar 하나를 2GB 미만 조각 2개로 나눠 올렸습니다.
- 빠진 것: 대회 데이터(`train.zip`·`valid.zip`·`test.zip`, `train/`·`valid/`·`test/`, `data/` 폴더, `*_image.png`)와 `.DS_Store`, `__pycache__`.

릴리스 파일 복원:

```bash
gh release download local-workspace-archive-20261022 -R CNUKoMaP/lhs
shasum -a 256 -c SHA256SUMS.txt
cat local_workspace_20261022.tar.part-* | tar -xf -
```

합친 tar의 SHA-256: `de607c3e9468fb51ea7037b6efe233d7289c04922dbd81700a47e73d5e3b7147`
