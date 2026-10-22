"""Build a local, server-free report. Run with conda run -n komap python ..."""
import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/model_results_1002_1"


def read(path):
    return json.loads(path.read_text())


def pct(value):
    return f"{100 * value:.6f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/model_results_1002_1_web_v1")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit(f"Existing output preserved: {out}. Choose a new --output path.")
    metrics = read(RESULT / "metrics.json")
    route = metrics["route"]
    run = Path(metrics["run"])
    tasks = read(run / "launcher_snapshot/pipeline_tasks.json")["tasks"]
    base = read(ROOT / "results/model_results_0930_1/metrics.json")
    classes = metrics["class_order"]
    rows = metrics["images"]
    assert len(rows) == 20
    assert abs(sum(r["image_miou"] for r in rows) / 20 - metrics["image_miou"]) < 1e-12
    for row in rows:
        assert (RESULT / "valid_mask_comparison" / (row["image"] + ".png")).is_file()
        assert abs(sum(row["class_iou"].values()) / 4 - row["image_miou"]) < 1e-12
    out.mkdir(parents=True)
    shutil.copytree(RESULT / "valid_mask_comparison", out / "valid_mask_comparison")
    manifest = {"source_result": str(RESULT), "source_run": str(run), "sha256": {}}
    for path in [RESULT / "metrics.json", RESULT / "ablation.json", *sorted((RESULT / "valid_mask_comparison").glob("*.png"))]:
        manifest["sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    colors = ["#194c99", "#4cb24c", "#997f4c", "#cccccc"]
    legend = "".join(f'<span><i style="background:{color}"></i>{name}</span>' for name, color in zip(classes, colors))
    gallery = []
    for n, row in enumerate(rows, 1):
        name = html.escape(row["image"])
        src = f"valid_mask_comparison/{name}.png"
        scores = "".join(f'<span>{c} <b>{pct(row["class_iou"][c])}</b></span>' for c in classes)
        gallery.append(f'''<article class="comparison" id="image-{n}">
<header><div><span class="eyebrow">VALID {n:02d} / 20</span><h3>{name}</h3></div>
<div class="image-score"><strong>{pct(row['image_miou'])}</strong><span>image mIoU</span><span>100% − image mIoU: {pct(1-row['image_miou'])}</span></div></header>
<div class="class-scores">{scores}</div>
<a href="{src}" target="_blank" rel="noopener" aria-label="{name} 비교 이미지 원본 크기로 열기"><img src="{src}" loading="{'eager' if n == 1 else 'lazy'}" decoding="async" alt="{name}: 좌상 원본, 우상 오류, 좌하 예측, 우하 정답"></a>
<footer>좌상 Original · 우상 Error · 좌하 Prediction · 우하 Ground truth <a href="{src}" target="_blank" rel="noopener">원본 크기로 보기</a></footer></article>''')
    route_rows = []
    for folder in ["model335_exhaustive_b4_routes", "model338_fine_gamma_route_search", "model339_joint_gamma_route_search", "model341_model329_joint_route_search", "model346_model336_into_341_route", "model349_model347_into_346_route"]:
        record = read(run / "outputs" / folder / "results.json")
        best = record.get("best_image_empty_zero", {})
        score = best.get("image_mean_miou", record.get("raw_best_valid_image_mean_miou_empty0"))
        label = folder.split("_")[0].replace("model", "Model")
        route_rows.append(f"<tr><th>{label}</th><td>{pct(score)}</td></tr>")
    gammas329 = read(run / "outputs/model341_model329_joint_route_search/results.json")["best_image_empty_zero"]["gamma329"]
    gamma_rows = "".join(f'<tr><th>{c}</th><td>{gammas329[i]:g}</td><td>{route["fixed_gamma336"][i]:g}</td><td>{route["raw_best_gamma347"][i]:g}</td></tr>' for i, c in enumerate(classes))
    task_rows = []
    for task in tasks:
        command = " ".join(task["argv"])
        task_rows.append(f'<tr><th>{html.escape(task["id"])}</th><td>{html.escape(", ".join(task["deps"]) or "독립 시작")}</td><td><code>{html.escape(command)}</code></td></tr>')
    mean_scores = "".join(f'<div><span>{c}</span><strong>{pct(sum(r["class_iou"][c] for r in rows)/20)}</strong></div>' for c in classes)
    cv = route["grouped_cv_vs_model346"]
    delta = (metrics["image_miou"] - base["image_miou"]) * 100
    content = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>1002 실험 보고서 · Model349</title><meta name="description" content="Model349 CE 클래스 가중치 제거 실험의 구조, 적용 사항, valid20 이미지 비교">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='5' fill='%23172237'/%3E%3Cpath d='M7 7h8v8H7z' fill='%23194c99'/%3E%3Cpath d='M17 7h8v8h-8z' fill='%234cb24c'/%3E%3Cpath d='M7 17h8v8H7z' fill='%23997f4c'/%3E%3Cpath d='M17 17h8v8h-8z' fill='%23ccc'/%3E%3C/svg%3E">
<style>
:root{font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif;color:#172237;background:#eef2f7;line-height:1.7;font-size:16px;scroll-behavior:smooth}*{box-sizing:border-box}body{margin:0}a{color:#164b9a;text-underline-offset:4px}a:focus-visible{outline:3px solid #ed9c36;outline-offset:5px}nav{background:#172237;color:#fff;padding:14px max(20px,calc((100% - 1200px)/2));display:flex;gap:24px;flex-wrap:wrap}nav a{color:#fff;font-size:.9rem}nav b{margin-right:auto}main{max-width:1200px;margin:auto;padding:32px 24px}h1{font-size:clamp(1.8rem,4vw,2.7rem);line-height:1.25;margin:8px 0 16px}h2{font-size:1.6rem;margin:0 0 18px}h3{margin:5px 0;font-size:1.15rem;overflow-wrap:anywhere}p{margin:12px 0}.eyebrow{font-size:.875rem;color:#526581;font-weight:700;letter-spacing:.06em}.intro{display:grid;grid-template-columns:1.7fr 1fr;gap:28px;margin-bottom:30px}.summary-score{background:#172237;color:#fff;padding:26px;border-radius:12px;display:flex;flex-direction:column;justify-content:center}.summary-score strong{font-size:2.5rem;line-height:1.3;color:#a4caff}.summary-score small{font-size:.875rem;color:#d0daea}section{background:#fff;border:1px solid #dbe2ec;border-radius:12px;padding:28px;margin:24px 0;scroll-margin-top:20px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.branch{border-top:4px solid #2455a2;background:#f4f7fc;padding:18px;border-radius:5px}.branch:nth-child(2){border-color:#36763e}.branch:nth-child(3){border-color:#8a6832}.branch code{font-size:.875rem}.flow{background:#172237;color:#eaf2ff;padding:20px;border-radius:8px;font-size:1rem;overflow-wrap:anywhere}.note{padding:12px 16px;border-left:4px solid #2455a2;background:#edf4ff}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.95rem}th,td{padding:12px;text-align:left;border-bottom:1px solid #e0e6ee;vertical-align:top}thead{background:#f2f5fa}code{overflow-wrap:anywhere;font-family:ui-monospace,monospace}.class-mean{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:20px}.class-mean div{padding:15px;background:#f3f6fb}.class-mean span,.class-mean strong{display:block}.legend{display:flex;flex-wrap:wrap;gap:18px;font-size:.95rem}.legend span{display:flex;gap:8px;align-items:center}.legend i{width:18px;height:18px;border:1px solid #8895a7;border-radius:3px}.comparison{background:#fff;border:1px solid #dbe2ec;border-radius:12px;overflow:hidden;margin:28px 0;scroll-margin-top:20px}.comparison header{display:flex;justify-content:space-between;gap:18px;padding:22px 24px;align-items:center}.image-score{text-align:right;display:flex;flex-direction:column;font-size:.9rem}.image-score strong{font-size:1.7rem;color:#164b9a}.class-scores{display:flex;flex-wrap:wrap;gap:12px 24px;padding:12px 24px;background:#f3f6fb;font-size:.875rem}.class-scores b{margin-left:8px}.comparison img{width:100%;height:auto;display:block;background:#fff}.comparison footer{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;padding:14px 24px;font-size:.875rem;color:#526581}details summary{cursor:pointer;font-weight:700;padding:10px 0}details code{font-size:.875rem}details td:last-child{min-width:420px}.sources{overflow-wrap:anywhere;font-size:.95rem}.top-link{display:block;text-align:right;margin:24px 0}.page-footer{color:#526581;font-size:.9rem;margin:30px 0}@media(max-width:760px){main{padding:22px 14px}.intro,.grid{grid-template-columns:1fr}.class-mean{grid-template-columns:repeat(2,1fr)}section{padding:20px}.comparison header{align-items:flex-start;flex-direction:column;padding:18px}.image-score{text-align:left}.summary-score strong{font-size:2.2rem}nav{gap:12px 18px;padding:14px}nav b{width:100%}th,td{padding:9px}}@media(prefers-reduced-motion:reduce){:root{scroll-behavior:auto}}@media print{nav,.top-link{display:none}section,.comparison{break-inside:avoid}body{background:white}main{max-width:none;padding:0}}
</style></head><body id="top"><nav aria-label="보고서 목차"><b>KoMaP / 1002</b><a href="#structure">모델 구조</a><a href="#applied">적용 사항</a><a href="#route">최종 조합</a><a href="#validation">검증 이미지 20장</a><a href="#sources">근거 자료</a></nav><main>
<div class="intro"><div><span class="eyebrow">2026.10.02 · MODEL_RESULTS_1002_1</span><h1>Model349<br>CE 클래스 가중치 제거 실험</h1><p>ResNet34·EfficientNet-B4·HRNet의 예측 확률을 클래스별로 조합합니다. 이전 학습 체크포인트를 재사용하지 않고, 모든 선행 모델부터 CE 클래스 가중치 없이 다시 학습했습니다.</p><a href="#validation">검증 비교 이미지로 이동</a></div><div class="summary-score"><span>Validation image mIoU</span><strong>@@SCORE@@</strong><span>20장 · 4개 클래스 고정 평균</span><small>100% − image mIoU: @@GAP@@</small></div></div>
<section id="structure"><h2>모델 구조와 학습 계보</h2><p>Model349는 실험·조합 번호입니다. 하나의 새 네트워크가 아니라 여러 학습 계열에서 만든 확률맵을 순차적으로 결합한 결과입니다. 인코더의 ImageNet 사전학습 초기화는 유지했습니다.</p><div class="grid">
<div class="branch"><h3>ResNet34 + U-Net++ / ASPP</h3><p>기본 분할 모델에 ASPP 다중 스케일 문맥을 추가하고, native patch·hard negative·EMA·표현 변형 계열을 학습합니다.</p><code>6 → 62 → 97 → 98 → 109<br>109 → restart51 / 59 / 61, 123, 181 → 183<br>109 → 194 → 195 → 239 → 242 → 257 → 269<br>195 → 220 (추론 시 TENT 적응)</code></div>
<div class="branch"><h3>EfficientNet-B4 + U-Net</h3><p>Model322·326·329를 각각 4개 grouped-CV fold로 학습합니다. 총 12개 fold, fold당 16,000 updates입니다.</p><p>322: gray3 / ImageNet 정규화 / scale 3 / area sampling.<br>326: zcdf3 / 전체 이미지 통계 / scale 3 / area sampling.<br>329: zcdf3 / 전체 이미지 통계 / target width 4000 / uniform sampling.</p></div>
<div class="branch"><h3>HRNet-W48 + U-Net++</h3><p>Model336은 고해상도 특징을 유지하는 인코더와 U-Net++ 디코더를 사용합니다. 학습용 경계 보조 헤드는 3×3 Conv로 경계를 예측하며, 추론에서는 분할 출력만 사용합니다.</p><p>336: 70 epochs, seed 0.<br>347: 이번 실행의 336 last.pt에서 12 epochs 미세조정, seed 347. 이 단계도 CE 클래스 가중치가 없습니다.</p></div></div><p class="flow">각 계열의 확률맵 → Model183 / native·full-stat 기반 route → 335 → 338 → 339 → 341 (+329) → 346 (+336) → 349 (+347) → 최종 4클래스 mask</p></section>
<section id="applied"><h2>이번 실험에서 적용한 사항</h2><div class="table-wrap"><table><thead><tr><th>항목</th><th>실제 설정</th></tr></thead><tbody>
<tr><th>변경한 요소</th><td>모든 semantic CE의 클래스 가중치를 제거했습니다. weight=None, 네 클래스에 동일한 가중치입니다. Model62는 loss-mode standard, Model347은 ce-class-weights 인자 없이 실행합니다.</td></tr>
<tr><th>손실과 샘플링</th><td>CE/Dice 비율, 클래스 평균 Dice, 경계 손실, 경계 BCE pos_weight, 희소 클래스·patch sampling은 유지했습니다. HRNet은 0.5 CE + 0.5 Dice에 경계 보조 손실(weight 1.0)을 더합니다.</td></tr>
<tr><th>데이터와 초기화</th><td>train 70장 / valid 20장. 데이터 fold·seed·epoch·batch 설정과 ImageNet 초기화를 유지했습니다. 이전 실험의 학습된 체크포인트는 사용하지 않았습니다.</td></tr>
<tr><th>HRNet 입력·최적화</th><td>patch 512, input scale 2.0, 이미지당 12 samples, batch 4 × accumulation 2. Model336 lr 3e-4, Model347 lr 1e-4 / encoder lr 배율 0.3, weight decay 1e-4.</td></tr>
<tr><th>추론과 후처리</th><td>기존 계열별 추론·TTA, 클래스별 gamma 탐색, 보정·후처리 경로를 유지했습니다. 예를 들어 257·269 계열은 D4 TTA를 사용하며 전체 모델의 TTA가 동일한 것은 아닙니다.</td></tr>
<tr><th>실행 구성</th><td>conda komap, 물리 GPU 6·7. 독립 모델·fold를 GPU당 하나씩 실행하는 총 59단계(50 GPU / 9 CPU) 파이프라인입니다.</td></tr>
</tbody></table></div><p class="note">이전 동일 파이프라인 결과(0930_1): @@BASE@@. 이번 결과와의 차이는 @@DELTA@@ %p입니다. 재학습을 포함한 단일 실행 비교이므로 이 차이만으로 가중치 제거 효과를 확정할 수 없습니다.</p></section>
<section id="route"><h2>실제로 선택된 클래스별 조합</h2><p>클래스 순서는 Al3Ni / Primary Si / Al / Eutectic Si입니다. Model339의 기본 조합에 Model329, Model336, Model347을 이 순서로 반영합니다. γ=0은 해당 추가 전문가를 반영하지 않고, γ=1은 해당 클래스 채널을 그 전문가 값으로 교체하는 설정입니다. 이후 정규화와 기존 후처리가 적용되므로 아래 값은 전체 모델의 단일 혼합 비율이 아닙니다.</p><div class="table-wrap"><table><thead><tr><th>클래스</th><th>329 γ (341 단계)</th><th>336 γ (346 단계)</th><th>347 γ (349 단계)</th></tr></thead><tbody>@@GAMMAS@@</tbody></table></div><p>마지막 단계는 γ ∈ {0, 0.25, 0.5, 0.75, 1}의 4클래스 조합 625개를 비교했습니다. 선택된 Model347은 Al 채널에만 γ=0.75로 반영됩니다.</p><details><summary>이번 실행의 단계별 image mIoU</summary><table><thead><tr><th>조합 단계</th><th>valid20 image mIoU</th></tr></thead><tbody>@@ROUTES@@</tbody></table></details><p class="note">valid 데이터는 체크포인트와 조합 선택에도 사용되므로 독립 테스트 점수가 아닙니다. 마지막 단계의 그룹별 교차 선택 비교(Model346 대비)는 @@CV@@ %p, bootstrap 95% 구간 @@CI@@ %p로 0을 포함합니다.</p></section>
<section><h2>평가 지표</h2><p>이미지별 mIoU = ¼ × Σ(4개 클래스의 TP / (TP + FP + FN))<br>최종 지표 = ¹⁄₂₀ × Σ(20장의 이미지별 mIoU)</p><p>정답과 예측에 모두 없는 클래스는 IoU=0입니다. 나머지 세 클래스가 각각 0.8이고 Primary Si가 모두 없다면 그 이미지의 mIoU는 60%입니다.</p><div class="class-mean">@@MEANS@@</div><p>위 수치는 각 클래스 IoU를 이미지 20장에 대해 평균한 값입니다. 100% − image mIoU는 픽셀 오류율이 아닙니다.</p></section>
<section id="validation"><h2>검증 이미지 비교 · 20장</h2><p>원래 보고서 순서대로 모든 비교 이미지를 이어 배치했습니다. 이미지에는 image mIoU와 100% − image mIoU가 표시되어 있습니다.</p><p>좌상: 원본 / 우상: 오류 위치 / 좌하: 예측 / 우하: 정답.<br>오류 그림의 흰색은 맞게 예측한 픽셀, 나머지는 해당 위치의 정답 색입니다. 초록색 오류는 실제 Primary Si를 다른 상으로 예측한 위치입니다.</p><div class="legend">@@LEGEND@@</div></section>
@@GALLERY@@
<section id="sources" class="sources"><h2>근거 자료와 실행 설정</h2><p>내용과 수치는 1002 결과의 metrics.json·ablation.json, 해당 실행의 source_snapshot·launcher_snapshot 및 단계별 results.json을 기준으로 정리했습니다.</p><p>원본 결과: <code>results/model_results_1002_1/</code><br>실행 스냅샷: <code>@@RUN@@</code></p><p>이미지는 기존 valid_mask_comparison 파일을 그대로 복사했습니다. 출처와 파일 SHA-256은 <a href="source_manifest.json">source_manifest.json</a>에 기록했습니다.</p><details><summary>59단계의 의존 관계와 저장된 실행 인자</summary><p>{source}는 실행의 source_snapshot, {run}은 위 실행 경로입니다.</p><div class="table-wrap"><table><thead><tr><th>단계</th><th>선행 단계</th><th>실행 인자</th></tr></thead><tbody>@@TASKS@@</tbody></table></div></details></section><a class="top-link" href="#top">맨 위로</a><p class="page-footer">KoMaP · 1002 실험 보고서 · 기존 실험 자료를 보존한 정적 HTML</p></main></body></html>'''
    substitutions = {"SCORE": pct(metrics["image_miou"]), "GAP": pct(1-metrics["image_miou"]), "BASE": pct(base["image_miou"]), "DELTA": f"{delta:+.6f}", "CV": f'{cv["image_mean_delta_pp"]:+.6f}', "CI": "[" + ", ".join(f"{v:+.6f}" for v in cv["group_bootstrap_95ci_delta_pp"]) + "]", "RUN": html.escape(str(run.relative_to(ROOT))), "GAMMAS": gamma_rows, "ROUTES": "".join(route_rows), "MEANS": mean_scores, "LEGEND": legend, "GALLERY": "\n".join(gallery), "TASKS": "".join(task_rows)}
    for key, value in substitutions.items():
        content = content.replace(f"@@{key}@@", value)
    assert "@@" not in content
    (out / "index.html").write_text(content, encoding="utf-8")
    (out / "README.txt").write_text("index.html을 브라우저로 직접 열면 됩니다. 서버나 학습 실행은 필요 없습니다.\n다른 컴퓨터로 이동할 때는 이 폴더 전체를 복사하세요.\n", encoding="utf-8")
    print(f"Created {out / 'index.html'} · {len(rows)} comparisons · {len(tasks)} pipeline stages")


if __name__ == "__main__":
    main()
