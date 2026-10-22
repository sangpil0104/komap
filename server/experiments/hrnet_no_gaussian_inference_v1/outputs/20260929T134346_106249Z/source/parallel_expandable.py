"""Separate allocator variant for full-image inference; same weights and arithmetic."""
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

from parallel_inference import main


if __name__=='__main__':main()
