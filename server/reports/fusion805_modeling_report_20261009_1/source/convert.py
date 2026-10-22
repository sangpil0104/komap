"""Convert with a local headless LibreOffice; no server or training environment edits."""
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
TOOLS=ROOT.parent/'.report_tools_20261009_1'
PROGRAM=TOOLS/'office/usr/lib/libreoffice/program'
env=dict(os.environ)
env['LD_LIBRARY_PATH']=f'{PROGRAM}:{TOOLS}/office/usr/lib/x86_64-linux-gnu'
env['FONTCONFIG_FILE']=str(TOOLS/'fonts.conf')
base=[str(PROGRAM/'soffice.bin'),f'-env:URE_BOOTSTRAP=vnd.sun.star.pathname:{PROGRAM}/fundamentalrc',
      f'-env:UserInstallation={TOOLS.as_uri()}/profile','--headless']
for ext,filter in [('doc','doc:MS Word 97'),('pdf','pdf')]:
    with (ROOT/f'validation/{ext}_conversion.log').open('w') as stream:
        command=base+['--convert-to',filter,'--outdir',str(ROOT),str(ROOT/'모델링_결과서_Fusion805.docx')]
        p=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if p.returncode==81:p=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(f'Office conversion failed: {ext}, exit {p.returncode}')
    assert (ROOT/f'모델링_결과서_Fusion805.{ext}').is_file()
dest=ROOT/'validation/doc_roundtrip';dest.mkdir(exist_ok=True)
with (ROOT/'validation/doc_roundtrip.log').open('w') as stream:
    subprocess.run(base+['--convert-to','pdf','--outdir',str(dest),str(ROOT/'모델링_결과서_Fusion805.doc')],
                   env=env,check=True,stdout=stream,stderr=subprocess.STDOUT)
print('DOC, PDF, and DOC re-opening PDF produced.')
