import time
from pathlib import Path
from alphase.engine.progress import Progress as OriginalProgress, duration
from hundredlab.workspace import read_json

class Progress(OriginalProgress):
    def __init__(self,run,cfg,world,elapsed=0):
        self.run,self.cfg=Path(run),cfg
        self.started,self.prior_elapsed=time.monotonic(),elapsed
        self.rates,self.validation_seconds,self.last={},0.,{}
        if elapsed and (self.run/'status.json').is_file():
            old=read_json(self.run/'status.json')
            self.rates={s['name']:s['seconds_per_update'] for s in old['stages'] if s.get('seconds_per_update')}
            self.validation_seconds=old.get('validation_seconds',0.)
