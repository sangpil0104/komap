"""User-run reproduction; --allow-after-deadline explicitly permits later reruns."""
import os
import sys

# Set before importing inference modules so spawned workers inherit the same policy.
if '--allow-after-deadline' in sys.argv:
    sys.argv.remove('--allow-after-deadline')
    os.environ['KOMAP_MANUAL_REPRODUCE_ALLOW_LATE']='1'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

import common
if os.environ.get('KOMAP_MANUAL_REPRODUCE_ALLOW_LATE')=='1':
    common.DEADLINE=float('inf')

from parallel_reproduce import main

if __name__=='__main__':
    if '--help' in sys.argv:
        print('Manual reproduction option: --allow-after-deadline permits reruns after the search cutoff.\n')
    main()
