import json
import math
import shutil
import time
from pathlib import Path

from alphase.utils import json_hash, write_json


class Progress:
    def __init__(self, run, cfg, world, elapsed=0):
        self.run, self.cfg = Path(run), cfg
        self.started, self.prior_elapsed = time.monotonic(), elapsed
        self.rates = {}
        self.validation_seconds = 0
        self.last = {}
        benchmark = Path("artifacts") / f"benchmark_{cfg['experiment']}_{world}gpu.json"
        if benchmark.exists():
            result = json.loads(benchmark.read_text())
            if result.get("config_hash") == json_hash(cfg):
                self.rates = {s["name"]: s["seconds_per_update"] for s in result["stages"]}
                self.validation_seconds = result.get("validation_seconds", 0)
        status = self.run / "status.json"
        if elapsed and status.exists():
            previous = json.loads(status.read_text())
            self.rates.update({s["name"]: s["seconds_per_update"] for s in previous.get("stages", []) if s.get("seconds_per_update")})
            self.validation_seconds = previous.get("validation_seconds", self.validation_seconds)
        shutil.copyfile(Path(__file__).with_name("dashboard.html"), self.run / "index.html")

    @property
    def elapsed(self):
        return self.prior_elapsed + time.monotonic()-self.started

    def write(self, stage_index, stage_step, update, state="training", seconds_per_update=None, **extra):
        stages = self.cfg["train"]["stages"]
        stage = stages[stage_index]
        if seconds_per_update is not None:
            old = self.rates.get(stage["name"], seconds_per_update)
            self.rates[stage["name"]] = .8*old+.2*seconds_per_update
        rate = self.rates.get(stage["name"])
        remaining = 0.
        approximate = False
        for i in range(stage_index, len(stages)):
            item = stages[i]
            steps = max(0, item["max_updates"]-(stage_step if i == stage_index else 0))
            item_rate = self.rates.get(item["name"])
            if item_rate is None and rate is not None:
                # Explicit provisional estimate until the future stage is benchmarked.
                def area(s):
                    return (sum(s.get("core_sizes", [1024])) / len(s.get("core_sizes", [1024]))+2*self.cfg["sampling"]["halo"])**2
                item_rate = rate * area(item)/area(stage)
                approximate = True
            if item_rate is None:
                remaining = None
                break
            remaining += steps*item_rate
            remaining += math.ceil(steps/self.cfg["train"]["eval_every_updates"])*self.validation_seconds
        total = sum(s["max_updates"] for s in stages)
        self.last.update(dict(state=state, experiment=self.cfg["experiment"], update=update,
                              total_updates=total, stage=stage["name"], stage_index=stage_index,
                              stage_step=stage_step, stage_total=stage["max_updates"],
                              elapsed_seconds=self.elapsed, remaining_seconds=remaining,
                              estimated_finish=time.time()+remaining if remaining is not None else None,
                              eta_provisional=approximate, seconds_per_update=rate,
                              validation_seconds=self.validation_seconds,
                              percent=100*update/total, updated_at=time.time(),
                              stages=[dict(name=s["name"], updates=s["max_updates"], seconds_per_update=self.rates.get(s["name"])) for s in stages]))
        self.last.update(extra)
        if state == "completed":
            self.last.update(remaining_seconds=0, percent=100)
        write_json(self.run/"status.json", self.last)


def duration(seconds):
    if seconds is None:
        return "measuring"
    minutes = int(seconds)//60
    return f"{minutes//60}h {minutes%60:02d}m"
