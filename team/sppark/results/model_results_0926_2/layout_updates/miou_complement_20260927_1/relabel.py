"""Replace comparison headers in place, with no backups of old pictures.

Only the upper-right header changes. Pixel panels and stored model metrics are
validated against saved originals, predictions and ground truth before replacement.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

OUTPUT = Path(__file__).resolve().parent
RESULT = OUTPUT.parents[1]
ROOT = RESULT.parents[1]
PANELS = RESULT / "valid_mask_comparison"
PALETTE = np.array([[153,127,76],[25,76,153],[76,178,76],[204,204,204]],np.uint8)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rgb(data):
    with Image.open(io.BytesIO(data)) as im:
        return np.array(im.convert("RGB"))


def labels(array):
    code=array.astype(np.int32)
    code=(code[...,0]<<16)|(code[...,1]<<8)|code[...,2]
    result=np.full(code.shape,255,np.uint8)
    for i,color in enumerate(PALETTE.astype(np.int32)):
        result[code==((color[0]<<16)|(color[1]<<8)|color[2])]=i
    assert not np.any(result==255),"Unknown mask palette"
    return result


def change_header(plan):
    seed,row,original,target,target_ids=plan
    relative=Path(f"seed_{seed:02d}")/row["image"]
    path=PANELS/relative
    prediction_path=RESULT/"predictions"/relative
    old_bytes=path.read_bytes()
    old=rgb(old_bytes)
    prediction_bytes=prediction_path.read_bytes()
    prediction=rgb(prediction_bytes)
    assert prediction.shape==target.shape==original.shape,relative
    pred_ids=labels(prediction)
    cm=np.bincount((4*target_ids.astype(np.int64)+pred_ids).ravel(),minlength=16).reshape(4,4)
    tp=cm.diagonal().astype(float)
    union=cm.sum(0)+cm.sum(1)-tp
    iou=np.divide(tp,union,out=np.ones(4),where=union>0)
    score=float(iou.mean())
    np.testing.assert_allclose(score,row["image_miou"],rtol=0,atol=1e-12)
    np.testing.assert_allclose(iou,row["class_iou"],rtol=0,atol=1e-12)
    assert np.flatnonzero(union==0).tolist()==row["absent_classes"]
    h,w=target_ids.shape
    margin,gap,header=16,24,76
    left,right,top,bottom=margin,margin+w+gap,margin,margin+header+h+gap
    assert old.shape==(bottom+header+h+margin,2*margin+2*w+gap,3),relative
    wrong=pred_ids!=target_ids
    errors=np.full_like(target,255)
    errors[wrong]=target[wrong]
    for array,x,y in [(original,left,top),(errors,right,top),(prediction,left,bottom),(target,right,bottom)]:
        np.testing.assert_array_equal(old[y+header:y+header+h,x:x+w],array)
    percent=score*100
    complement=100-percent
    first=f"image mIoU: {percent:.4f}%"
    second=f"100% - image mIoU: {complement:.4f}%"
    heading=Image.new("RGB",(w,header),"white")
    draw=ImageDraw.Draw(heading)
    font=ImageFont.truetype(FONT,28)
    for y,text in [(4,first),(42,second)]:
        assert draw.textlength(text,font=font)<=w,(relative,text)
        draw.text((0,y),text,fill="black",font=font,anchor="lt")
        assert draw.textbbox((0,y),text,font=font,anchor="lt")[3]<=header
    canvas=Image.fromarray(old)
    canvas.paste(heading,(right,top))
    expected=old.copy()
    expected[top:top+header,right:right+w]=np.array(heading)
    temporary=path.with_name(path.name+".miou-new.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        canvas.save(temporary,format="PNG",compress_level=3)
        new_bytes=temporary.read_bytes()
        saved=rgb(new_bytes)
        # Exact whole-image comparison proves that no pixels outside the header changed.
        np.testing.assert_array_equal(saved,expected)
        assert digest(path.read_bytes())==digest(old_bytes),"Picture changed concurrently"
        temporary.replace(path)  # User explicitly requested replacement without a backup.
        assert digest(path.read_bytes())==digest(new_bytes)
    finally:
        if temporary.exists():temporary.unlink()
    record=dict(seed=seed,image=row["image"],image_miou_percent=percent,
                one_hundred_minus_image_miou_percent=complement,
                image_miou_display=f"{percent:.4f}%",complement_display=f"{complement:.4f}%",
                header_box=[right,top,right+w,top+header],
                panel_pixels_unchanged=True,metrics_verified=True,sha256=digest(new_bytes))
    return record,{str(prediction_path.relative_to(ROOT)):digest(prediction_bytes)}


def main():
    started=time.monotonic()
    if (OUTPUT/"verification.json").exists():
        raise FileExistsError("This update is already complete")
    suite=json.loads((RESULT/"suite.json").read_text())
    assert suite["state"]=="completed"
    metrics={}
    protected={}
    checkpoint_stats={}
    for p in sorted((RESULT/"metrics").glob("seed_*.json")):
        data=json.loads(p.read_text())
        metrics[data["seed"]]=data
        protected[str(p.relative_to(ROOT))]=digest(p.read_bytes())
        checkpoint=Path(data["checkpoint"])
        st=checkpoint.stat()
        checkpoint_stats[str(checkpoint)]=[st.st_size,st.st_mtime_ns]
    assert set(metrics)==set(range(100))
    for p in [RESULT/"suite.json",RESULT/"seed_results.csv"]:
        protected[str(p.relative_to(ROOT))]=digest(p.read_bytes())
    names=sorted(r["image"] for r in metrics[0]["per_image"])
    assert len(names)==20 and len(set(names))==20
    for data in metrics.values():
        assert sorted(r["image"] for r in data["per_image"])==names
    expected={Path(f"seed_{s:02d}")/n for s in metrics for n in names}
    assert {p.relative_to(PANELS) for p in PANELS.glob("seed_*/*.png")}==expected
    previous_history={p.relative_to(RESULT).as_posix():digest(p.read_bytes())
                      for p in (RESULT/"comparison_history").rglob("*") if p.is_file()}
    arrays={}
    for name in names:
        image_path=ROOT/"dataset/valid/images"/name
        target_path=ROOT/"dataset/valid/masks"/name.replace("_image.png","_mask.png")
        image_bytes,target_bytes=image_path.read_bytes(),target_path.read_bytes()
        original,target=rgb(image_bytes),rgb(target_bytes)
        arrays[name]=(original,target,labels(target))
        protected[str(image_path.relative_to(ROOT))]=digest(image_bytes)
        protected[str(target_path.relative_to(ROOT))]=digest(target_bytes)
    plans=[(s,row,*arrays[row["image"]]) for s,data in sorted(metrics.items()) for row in data["per_image"]]
    # Preserve only documentation when updating the explanatory text. No old image is copied.
    report=RESULT/f"{RESULT.name}.md"
    readme=PANELS/"README.md"
    for source,dest in [(report,OUTPUT/"previous_report.md"),(readme,OUTPUT/"previous_readme.md")]:
        if dest.exists():
            assert dest.read_bytes()==source.read_bytes()
        else:
            shutil.copy2(source,dest)
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record,hashes in tqdm(pool.map(change_header,plans),total=len(plans),
                                  desc="Replace mIoU headers (no image backup)",unit="image",mininterval=1):
            records.append(record)
            protected.update(hashes)
    before="- 우측 상단 error rate = 틀린 픽셀 수 / 전체 픽셀 수 × 100. 제목 28px, 패널 간격 24px."
    after="- 우측 상단에는 해당 이미지의 **image mIoU(%)**와 **100% − image mIoU(%)**를 소수점 넷째 자리까지 표시합니다. 두 줄 모두 28px, 패널 간격 24px."
    text=report.read_text()
    assert before in text
    text=text.replace(before,after)
    relative=OUTPUT.relative_to(RESULT).as_posix()
    text=text.replace(after,after+f"\n- 현재 표시 기준과 2,000장 검증: [{relative}/verification.json]({relative}/verification.json). 이번 비교 사진 교체는 사용자 요청으로 이미지 백업 없이 수행했습니다.")
    report.write_text(text)
    readme.write_text(
        "# 비교 이미지 표시\n\n"
        "- 좌상 Original image / 우상 오류 위치 / 좌하 predicted mask / 우하 ground-truth mask.\n"
        "- 우측 상단 두 줄: image mIoU(%)와 100% − image mIoU(%). 소수점 넷째 자리까지 표시합니다.\n"
        "- image mIoU는 해당 이미지의 4개 클래스 IoU 평균이며 정답·예측 양쪽에 없는 클래스는 IoU=1입니다.\n"
        "- 오류 위치는 정답 색, 맞게 예측한 위치는 흰색입니다.\n"
        "- 두 수치의 글자 크기는 28px, 패널 간격은 24px입니다. 이미지와 마스크는 원해상도 그대로입니다.\n"
        "- 이번 변경은 사용자 요청에 따라 기존 비교 사진을 백업 없이 덮어썼습니다.\n"
        f"- [표시 수치 CSV](../{relative}/display_metrics.csv), [검증 기록](../{relative}/verification.json).\n")
    with (OUTPUT/"display_metrics.csv").open("x",newline="") as f:
        fields=["seed","image","image_miou_percent","one_hundred_minus_image_miou_percent","image_miou_display","complement_display"]
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(records)
    for path,before_hash in tqdm(protected.items(),desc="Verify masks and metrics unchanged",unit="file",mininterval=1):
        assert digest((ROOT/path).read_bytes())==before_hash,path
    for path,stats in checkpoint_stats.items():
        st=Path(path).stat()
        assert [st.st_size,st.st_mtime_ns]==stats,path
    assert {p.relative_to(RESULT).as_posix():digest(p.read_bytes())
            for p in (RESULT/"comparison_history").rglob("*") if p.is_file()}==previous_history
    assert len(list(PANELS.glob("seed_*/*.png")))==2000
    assert not list(PANELS.rglob("*.miou-new.tmp"))
    payload=dict(status="passed",updated_at=datetime.now(timezone.utc).isoformat(),images=2000,seeds=100,
                 backup_of_replaced_images_created=False,existing_history_unchanged=True,
                 changed_region="upper-right header only",header_font_px=28,decimal_places=4,
                 score_policy="mean of four class IoUs for this image; empty union = 1",
                 second_value="100 - image_miou_percent",panels_pixel_verified=2000,
                 metrics_recomputed_and_verified=2000,protected_files_sha256=protected,
                 checkpoint_stats=checkpoint_stats,script_sha256=digest(Path(__file__).read_bytes()),
                 elapsed_seconds=time.monotonic()-started,records=records)
    with (OUTPUT/"verification.json").open("x") as f:
        json.dump(payload,f,ensure_ascii=False,indent=2);f.write("\n")
    print(json.dumps(dict(status="passed",images=len(records),elapsed_seconds=time.monotonic()-started,
                         image_backup_created=False,result=str(PANELS)),indent=2))


if __name__=="__main__":
    main()
