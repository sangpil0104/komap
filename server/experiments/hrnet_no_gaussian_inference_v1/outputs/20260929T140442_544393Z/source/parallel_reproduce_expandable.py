import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

from parallel_reproduce import main

if __name__=='__main__':main()
