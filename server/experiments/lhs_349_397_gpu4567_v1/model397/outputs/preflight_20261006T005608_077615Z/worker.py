import subprocess,sys,os
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ['PYTHON']=sys.executable
subprocess.run(['bash',str(root/'fold.sh'),*sys.argv[1:]],check=True)
