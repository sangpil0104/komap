"""Apply the current native-resolution 2x2 layout to saved Multi-Otsu results.

Preserves the previous report and comparison directory before installation.
Does not change the model, evaluation source, predictions, or original metrics.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

HISTORY = Path(__file__).resolve().parent
RESULT = HISTORY.parents[1]
ROOT = RESULT.parents[1]
STAGE = HISTORY / "new_valid_mask_comparison"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def labels(array, palette):
    ids = np.full(array.shape[:2], 255, dtype=np.uint8)
    for i, color in enumerate(palette):
        ids[np.all(array == color, axis=-1)] = i
    assert not np.any(ids == 255), "Unknown palette color"
    return ids


def render(row, palette):
    name = row["image_id"]
    original = rgb(ROOT / "dataset/valid/images" / f"{name}_image.png")
    target = rgb(ROOT / "dataset/valid/masks" / f"{name}_mask.png")
    prediction = rgb(RESULT / "masks" / f"{name}.png")
    assert original.shape == target.shape == prediction.shape
    gt, pred = labels(target,palette), labels(prediction,palette)
    cm = np.bincount((4*gt.astype(np.int64)+pred).ravel(),minlength=16).reshape(4,4)
    np.testing.assert_array_equal(cm,row["confusion_matrix"])
    tp = cm.diagonal().astype(float)
    union = cm.sum(0)+cm.sum(1)-tp
    iou = np.divide(tp,union,out=np.full(4,np.nan),where=union>0)
    scores = {"one":np.nan_to_num(iou,nan=1).mean()*100,
              "zero":np.nan_to_num(iou,nan=0).mean()*100,"exclude":np.nanmean(iou)*100}
    for policy,score in scores.items():
        assert abs(score-row[f"image_miou_{policy}"]) < 1e-10
    wrong = gt != pred
    error_pixels, pixels = int(wrong.sum()), int(wrong.size)
    error_rate = error_pixels/pixels*100
    error = np.full_like(target,255)
    error[wrong] = target[wrong]
    h,w = gt.shape
    margin,gap,header = 16,24,76
    left,right,top,bottom = margin,margin+w+gap,margin,margin+header+h+gap
    canvas = Image.new("RGB",(2*margin+2*w+gap,bottom+header+h+margin),"white")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",28)
    detail_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",20)
    panels = [
        ("original",original,left,top,"Original image",f"{name} | Multi-Otsu"),
        ("errors",error,right,top,f"Errors | error rate: {error_rate:.2f}%",
         "Wrong pixels / all pixels | GT colors; correct: white"),
        ("predicted",prediction,left,bottom,"Predicted mask",f"image mIoU: {scores['one']:.4f}%"),
        ("ground_truth",target,right,bottom,"Ground-truth mask","Primary Si / Al3Ni / Eutectic Si / Al"),
    ]
    boxes={}
    for key,array,x,y,title,subtitle in panels:
        assert draw.textlength(title,font=title_font)<=w
        assert draw.textlength(subtitle,font=detail_font)<=w
        draw.text((x,y+4),title,font=title_font,fill="black",anchor="lt")
        draw.text((x,y+42),subtitle,font=detail_font,fill="#444444",anchor="lt")
        canvas.paste(Image.fromarray(array),(x,y+header))
        boxes[key]=[x,y+header,x+w,y+header+h]
    path=STAGE/f"{name}.png"
    canvas.save(path)
    saved=rgb(path)
    np.testing.assert_array_equal(saved,np.array(canvas))
    for key,array,x,y,_,_ in panels:
        np.testing.assert_array_equal(saved[y+header:y+header+h,x:x+w],array)
    assert np.all(saved[:,left+w:right]==255)
    assert np.all(saved[top+header+h:bottom]==255)
    return dict(image_id=name,error_pixels=error_pixels,total_pixels=pixels,
                error_rate_percent=error_rate,error_rate_display=f"{error_rate:.2f}%",
                image_miou_one=float(scores["one"]),panel_boxes=boxes,
                sha256=digest(path),pixel_verification="passed")


def main():
    started=time.monotonic()
    STAGE.mkdir(exist_ok=False)
    report=RESULT/f"{RESULT.name}.md"
    original_text=report.read_text()
    original_hashes={p.relative_to(RESULT).as_posix():digest(p) for p in RESULT.rglob("*")
                     if p.is_file() and "comparison_history" not in p.relative_to(RESULT).parts}
    rows=json.loads((RESULT/"per_image.json").read_text())
    assert len(rows)==20 and len({r["image_id"] for r in rows})==20
    assert {p.stem for p in (RESULT/"valid_mask_comparison").glob("*.png")}=={r["image_id"] for r in rows}
    palette=np.array(json.loads((RESULT/"model.json").read_text())["palette"],dtype=np.uint8)
    data_paths=[ROOT/"dataset/valid"/kind/f"{r['image_id']}_{suffix}.png"
                for r in rows for kind,suffix in [("images","image"),("masks","mask")]]
    data_hashes={str(p.relative_to(ROOT)):digest(p) for p in data_paths}
    records=[render(r,palette) for r in tqdm(rows,desc="Reformat Multi-Otsu comparisons",unit="image")]
    assert len(list(STAGE.glob("*.png")))==20
    summary=json.loads((RESULT/"summary.json").read_text())
    assert abs(np.mean([r["image_miou_one"] for r in records])-summary["image_miou_one"])<1e-10
    with (HISTORY/"error_rates.csv").open("x",newline="") as f:
        fields=["image_id","error_pixels","total_pixels","error_rate_percent","error_rate_display","image_miou_one"]
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(records)
    rates={r["image_id"]:r["error_rate_percent"] for r in records}
    text=original_text.replace(
        "| 임계값 계산 | image mIoU (%) | 비교 그림 | 히스토그램 |",
        "| 임계값 계산 | image mIoU (%) | error rate (%) | 비교 그림 | 히스토그램 |")
    text=text.replace("|---|---:|---:|---:|---|---:|---|---|",
                      "|---|---:|---:|---:|---|---:|---:|---|---|")
    lines=[]
    for line in text.splitlines():
        fields=[part.strip() for part in line.split("|")]
        if len(fields)==10 and fields[1] in rates:
            fields.insert(7,f"{rates[fields[1]]:.4f}")
            line="| "+" | ".join(fields[1:-1])+" |"
        lines.append(line)
    history_rel=HISTORY.relative_to(RESULT).as_posix()
    text="\n".join(lines)+"\n"
    before="좌측 상단: 원본, 우측 상단: 맞는 픽셀은 흰색·틀린 픽셀은 정답 상의 색, 좌측 하단: 예측, 우측 하단: 정답입니다. 각 패널은 원본 해상도로 저장하고 저장된 픽셀을 재검사합니다."
    after=before+"\n\n우측 상단 제목에 **error rate = 틀린 픽셀 수 / 전체 픽셀 수 × 100**을 표시합니다. 최근 결과와 동일하게 제목 28px, 부제목 20px, 패널 간격 24px, 여백 16px를 적용했습니다. 원본·예측·정답은 축소하지 않았습니다.\n\n"+f"[표시 변경 검증 기록]({history_rel}/layout.json) · [오류율 CSV]({history_rel}/error_rates.csv) · [기존 비교 사진]({history_rel}/valid_mask_comparison/) · [기존 보고서]({history_rel}/{report.name})"
    assert before in text
    text=text.replace(before,after)
    text=text.replace("[그림 픽셀 검증](verification.json)","[최초 평가의 그림 검증](verification.json)")
    stage_report=HISTORY/"new_report.md"
    stage_report.write_text(text)
    (STAGE/"README.md").write_text("# Multi-Otsu 검증 비교 사진\n\n좌상 원본 / 우상 오류 위치 및 error rate / 좌하 예측 / 우하 정답.\n\n정답 픽셀은 흰색, 오류 픽셀은 해당 위치의 정답 색입니다. error rate는 전체 픽셀 중 틀린 픽셀 비율(%)입니다.\n\n제목 28px, 부제목 20px, 간격 24px, 여백 16px. 모든 패널은 원해상도입니다.\n")
    for rel,before_hash in original_hashes.items():
        assert digest(RESULT/rel)==before_hash,rel
    for rel,before_hash in data_hashes.items():
        assert digest(ROOT/rel)==before_hash,rel
    shutil.copy2(report,HISTORY/report.name)
    current=RESULT/"valid_mask_comparison"
    current.rename(HISTORY/"valid_mask_comparison")
    try:
        STAGE.rename(current)
        stage_report.replace(report)
    except BaseException:
        if current.exists():current.rename(STAGE)
        (HISTORY/"valid_mask_comparison").rename(current)
        shutil.copy2(HISTORY/report.name,report)
        raise
    for rel,before_hash in original_hashes.items():
        if rel.startswith("valid_mask_comparison/") or rel==report.name:
            assert digest(HISTORY/rel)==before_hash,rel
        else:
            assert digest(RESULT/rel)==before_hash,rel
    layout=dict(status="passed",created_at=datetime.now(timezone.utc).isoformat(),images=records,
                image_count=20,title_font_px=28,detail_font_px=20,gap_px=24,margin_px=16,header_px=76,
                resized=False,error_rate_definition="100 * wrong_pixels / total_pixels",
                original_result_sha256=original_hashes,input_sha256=data_hashes,
                old_comparisons_preserved=True,model_and_metrics_unchanged=True,
                script_sha256=digest(Path(__file__)),elapsed_seconds=time.monotonic()-started)
    with (HISTORY/"layout.json").open("x") as f:
        json.dump(layout,f,ensure_ascii=False,indent=2);f.write("\n")
    print(f"Updated {len(records)} comparisons: {current}")
    print(f"Original report and figures: {HISTORY}")
    print("Pixel checks, confusion matrices, all three mIoUs and preservation checks passed")


if __name__=="__main__":
    main()
