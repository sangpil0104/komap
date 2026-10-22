import argparse

from alphase.engine.trainer import train
from alphase.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train native-resolution CNN or DINOv3 hybrid segmentation")
    parser.add_argument("--config", default="configs/cnn_unet.yaml")
    parser.add_argument("--split", default="official")
    parser.add_argument("--run-dir", "--run", dest="run", help="Output directory (--run-dir is compatible with torchrun)")
    parser.add_argument("--resume", help="Trusted local last.pt from this run")
    parser.add_argument("--overwrite", action="store_true", help="Archive the old run and start again")
    parser.add_argument("--stop-after", type=int, help="Pause after this global optimizer update (smoke/resume check)")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    train(load_config(args.config, args.set), args.split, args.run, args.resume, args.overwrite, args.stop_after)


if __name__ == "__main__":
    main()
