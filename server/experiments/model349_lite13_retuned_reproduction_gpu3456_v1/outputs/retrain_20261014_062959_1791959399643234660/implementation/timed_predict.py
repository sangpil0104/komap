"""Measure fresh full-pipeline inference through subprocess completion, including disk writes."""
from pathlib import Path
import argparse, os, subprocess, time
from common import read, write, now, devices

SCOPE = ('Fresh end-to-end wall time: code/weight link preparation, Python startup, model loading, '
         'image I/O and preprocessing, R inference/TTA/TENT, E inference, CPU F/postprocessing, '
         'probability and mask writes, temporary cleanup. Excludes training, GT scoring and report figures. '
         'No explicit warm-up; no precomputed predictions reused. Child exit follows CPU mask writes '
         'and therefore completed CUDA work. Seconds/image = batch wall time divided by image count; '
         'not an isolated single-image latency or sum of GPU compute times.')

def run_prediction(source,weights,images,output,gpus,split):
    source,weights,images,output=map(Path,[source,weights,images,output])
    if output.exists():raise FileExistsError(output)
    count=len(list(images.glob('*_image.png')))
    if count!={'valid':20,'test':10}[split]:raise ValueError(f'Unexpected {split} count: {count}')
    started=now();env=dict(os.environ,ALLOW_RETRAINED='1')
    command=['bash',str(source/'run_predict.sh'),str(weights),str(images),str(output),gpus]
    hardware=devices();t=time.perf_counter()
    subprocess.run(command,env=env,check=True)
    seconds=time.perf_counter()-t
    names={p.stem.removesuffix('_image') for p in images.glob('*_image.png')}
    for d in ['predicted_labels','predicted_masks']:
        actual={p.stem for p in (output/'fusion'/d).glob('*.png')}
        if actual!=names:raise ValueError(f'Incomplete prediction inventory: {d}')
    if not (output/'completion.json').is_file():raise RuntimeError('Missing prediction completion receipt')
    row=dict(split=split,images=count,wall_seconds=seconds,seconds_per_image=seconds/count,
             images_per_second=count/seconds,started_at=started,finished_at=now(),scope=SCOPE,
             requested_physical_gpus=[3,4,5,6],gpu_uuids=gpus.split(','),hardware=hardware,
             execution='R components across GPUs 3-6; E on first visible GPU (3); F on CPU',
             phase_times=read(output/'phase_timings.json'),command=command)
    write(output/'inference_timing.json',row)
    print(f'INFERENCE {split}: {count} images, {seconds:.3f}s total, {seconds/count:.3f}s/image',flush=True)
    return row

def main():
    p=argparse.ArgumentParser()
    for key in ['source','weights','images','output','gpus','split']:p.add_argument('--'+key,required=True)
    a=p.parse_args();run_prediction(**vars(a))
if __name__=='__main__':main()
