import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from alphase.utils import write_json
from lab.workspace import BASELINE, WORKSPACE, load_experiment, output_path, plan


def estimated_seconds(gpus):
    path = BASELINE/"artifacts"/f"benchmark_cnn_unet_{gpus}gpu.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    rates = [s["seconds_per_update"] for s in data["stages"]]
    result = {}
    for name in plan()["default_suite"]:
        cfg = load_experiment(name)
        total = sum(stage["max_updates"]*rate for stage, rate in zip(cfg["train"]["stages"], rates))
        checks = sum((s["max_updates"]+249)//250 for s in cfg["train"]["stages"])
        result[name] = total + checks*data["validation_seconds"]
    return result


class SuiteHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
    def do_GET(self):
        path = unquote(urlparse(self.path).path).lstrip("/")
        if not path:
            self.path = "/suite.html"
            path = "suite.html"
        allowed = path in ("suite.html", "suite_status.json", "reports/comparison.html", "reports/comparison.csv",
                           "reports/per_image_comparison.csv", "reports/comparison.json")
        allowed |= bool(re.fullmatch(r"runs/(fullframe|boundary|seed2027|seed2028|dino_large)/(official|all90|fold[0-4])/(index.html|status.json|metrics.jsonl|validation/(raw|ema)/preview_[1-3]\.png)", path))
        allowed |= bool(re.fullmatch(r"reports/reference_baseline_(plain|d4)\.csv", path))
        allowed |= bool(re.fullmatch(r"evaluations/[A-Za-z0-9_-]+/official/valid/[A-Za-z0-9_-]+/per_image_metrics\.csv", path))
        allowed |= bool(re.fullmatch(r"ensembles/[A-Za-z0-9_-]+/official/valid/per_image_metrics\.csv", path))
        if not allowed:
            self.send_error(404)
            return
        super().do_GET()


def monitor_suite(port):
    document = Path(__file__).with_name("suite_dashboard.html").read_text()
    output_path("suite.html").write_text(document)
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(SuiteHandler, directory=str(output_path())))
    print(f"Whole experiment suite: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
