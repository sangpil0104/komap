import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
    def do_GET(self):
        # Expose dashboard artifacts only; checkpoints and dataset paths stay private.
        from urllib.parse import unquote, urlparse
        path = unquote(urlparse(self.path).path).lstrip("/")
        allowed = path in ("", "index.html", "status.json", "metrics.jsonl")
        allowed |= path in {f"validation/{kind}/preview_{i}.png" for kind in ("raw", "ema") for i in range(1, 4)}
        if not allowed:
            self.send_error(404)
            return
        super().do_GET()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/cnn_unet/official")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    root = Path(args.run).resolve()
    if not (root/"resolved_config.yaml").is_file():
        raise SystemExit("Start training first, then launch this monitor for its run directory.")
    shutil.copyfile(Path(__file__).parents[1]/"engine/dashboard.html", root/"index.html")
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(root)))
    print(f"학습 모니터: http://{args.host}:{args.port} (원격 서버에서는 VS Code 포트 전달 사용)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
