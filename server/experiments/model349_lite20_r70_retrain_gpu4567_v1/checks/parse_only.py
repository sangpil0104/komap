"""Stop at argument parsing, before an entrypoint can construct/train a model."""
import argparse
import json
from pathlib import Path
import runpy
import sys

script = Path(sys.argv[1])
output = Path(sys.argv[2])
arguments = sys.argv[3:]
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), *arguments]
original = argparse.ArgumentParser.parse_args


class ParsedOnly(BaseException):
    pass


def intercept(self, *args, **kwargs):
    parsed = original(self, *args, **kwargs)
    output.write_text(json.dumps(vars(parsed), indent=2, default=str) + '\n')
    raise ParsedOnly()


argparse.ArgumentParser.parse_args = intercept
try:
    runpy.run_path(str(script), run_name='__main__')
except ParsedOnly:
    print(f'PARSED ONLY: {script.name}; no training function was entered.')
else:
    raise RuntimeError('Entrypoint did not reach argument parsing.')
