"""OOF member probabilities for lite19a (run inside oof_design/): published B4 fold models are linked as the fold's B4
members, high-res members are restart51 and restart61 only (lite19a uses restart61 for every image).
    python predict_oof_l19a.py --work WORK --fold K --b4-root WEIGHTS/R"""
import sys, argparse
from pathlib import Path
import members as M
M.HIGH = {'restart51': 'restart51', 'pairwise': 'restart61'}
import predict_oof
B4DIR = {'b4_322': 'model322_team_b4_group_oof', 'b4_326': 'model326_hybrid_fullstats_group_oof', 'b4_329': 'model329_fullstats_width4000_uniform_group_oof'}
ap = argparse.ArgumentParser(); ap.add_argument('--work', type=Path, required=True); ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--b4-root', type=Path, required=True); a, rest = ap.parse_known_args()
run = a.work / 'runs' / f'fold{a.fold}' / 'seed0'
for k, d in B4DIR.items():
    link = run / k
    if not link.exists(): link.symlink_to(a.b4_root / d / f'fold{a.fold}')
sys.argv = [sys.argv[0], '--work', str(a.work), '--fold', str(a.fold), '--seeds', '0'] + rest
predict_oof.main()
