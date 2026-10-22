"""Validate delivered file formats, numbers, links, embedded figures and protected inputs."""
from pathlib import Path
import hashlib,json,re,zipfile
import pymupdf as fitz
from PIL import Image,ImageDraw
from docx import Document
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
md=(ROOT/'모델링_결과서_Fusion805.md').read_text()
docx=ROOT/'모델링_결과서_Fusion805.docx';doc=ROOT/'모델링_결과서_Fusion805.doc'
assert doc.read_bytes()[:8]==bytes.fromhex('d0cf11e0a1b11ae1'),'Not genuine binary DOC'
assert zipfile.is_zipfile(docx)
native=Document(docx);counts=read(ROOT/'validation/manuscript_counts.json')
assert len(native.tables)==counts['tables']+20
assert len(native.inline_shapes)==counts['main_figures']+40
assert len(list((ROOT/'tables').glob('[0-9]*.png')))==24
assert len(list((ROOT/'tables').glob('[0-9]*.csv')))==24
assert len(list((ROOT/'figures').glob('*.png')))==23
assert len(list((ROOT/'figures').glob('*.svg')))==23
for p in [*ROOT.glob('figures/*.png'),*ROOT.glob('tables/*.png'),*ROOT.glob('appendix/*/*.png')]:
    with Image.open(p) as im:im.verify()
images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',md)
assert len(images)==63 and all((ROOT/p).is_file() for p in images)
refs=read(ROOT/'source/manuscript.json')['references']
assert all(v['url'].startswith('https://') or (ROOT/v['url']).exists() for v in refs.values())
for name,digest in read(ROOT/'evidence/protected_input_hashes.json').items():assert sha(PROJECT/name)==digest,name
pdf=fitz.open(ROOT/'모델링_결과서_Fusion805.pdf');roundtrip=fitz.open(ROOT/'validation/doc_roundtrip/모델링_결과서_Fusion805.pdf')
layout=[]
for label,document in [('docx_pdf',pdf),('doc_pdf',roundtrip)]:
    full='\n'.join(page.get_text() for page in document);compact=re.sub(r'\s+','',full)
    for required in ['80.236507','80.267268','1-1.','1-2.','1-3.','2-1.','2-2.','3-1.','3-2.','3-3.','미제공','test10']:
        assert required in compact,(label,required)
    assert '\ufffd' not in full,(label,'replacement character')
    assert sum(len(page.get_image_info()) for page in document)==63,(label,'embedded image count')
    assert sum('부록A-' in re.sub(r'\s+','',page.get_text()) for page in document)==20
    for i,page in enumerate(document):
        text=page.get_text();assert len(text.strip())>60,(label,'unexpected blank page',i+1)
        for img in page.get_image_info():
            rect=fitz.Rect(img['bbox']);assert rect.x0>=-1 and rect.y0>=-1 and rect.x1<=page.rect.width+1 and rect.y1<=page.rect.height+1,(label,i,'image clipping')
        layout.append(dict(format=label,page=i+1,chars=len(text),images=len(page.get_image_info()),width=page.rect.width,height=page.rect.height))
    (ROOT/f'validation/{label}_text.txt').write_text(full)
indices=sorted(set([0,1,7,10,11,12,14,15,20,21,24,26,29,len(pdf)-20,len(pdf)-1]))
canvas=Image.new('RGB',(1500,((len(indices)+2)//3)*590),'#d9dde2');draw=ImageDraw.Draw(canvas)
for n,i in enumerate(indices):
    pix=pdf[i].get_pixmap(matrix=fitz.Matrix(.85,.85));im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im.thumbnail((480,550))
    x=(n%3)*500+(500-im.width)//2;y=(n//3)*590;canvas.paste(im,(x,y));draw.text(((n%3)*500+12,y+558),f'Page {i+1}',fill='black')
canvas.save(ROOT/'validation/document_contact_sheet.jpg')
figures=sorted((ROOT/'figures').glob('*.png'))
sheet=Image.new('RGB',(1500,((len(figures)+2)//3)*360),'#eeeeee');draw=ImageDraw.Draw(sheet)
for n,path in enumerate(figures):
    im=Image.open(path).convert('RGB');im.thumbnail((480,320));x=(n%3)*500+(500-im.width)//2;y=(n//3)*360+12
    sheet.paste(im,(x,y));draw.text(((n%3)*500+12,(n//3)*360+338),path.stem,fill='black')
sheet.save(ROOT/'validation/figures_contact_sheet.jpg')
for i in [0,11,14,len(pdf)-20]:pdf[i].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(ROOT/f'validation/page_{i+1:02d}.png')
(ROOT/'validation/pdf_layout.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2)+'\n')
report=dict(ok=True,doc_binary_verified=True,docx_native_tables=24,docx_appendix_tables=20,embedded_images=63,
    main_figures=23,tables=24,appendix_panels=40,pdf_pages=len(pdf),doc_roundtrip_pages=len(roundtrip),
    doc_reopened_successfully=True,all_local_report_links_valid=True,all_images_decode=True,no_image_clipping=True,
    protected_inputs_unchanged=len(read(ROOT/'evidence/protected_input_hashes.json')),numeric_audit=read(ROOT/'validation/numeric_audit.json'))
report['document_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [doc,docx,ROOT/'모델링_결과서_Fusion805.pdf',ROOT/'모델링_결과서_Fusion805.md',ROOT/'validation/doc_roundtrip/모델링_결과서_Fusion805.pdf']}
(ROOT/'validation/final_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
