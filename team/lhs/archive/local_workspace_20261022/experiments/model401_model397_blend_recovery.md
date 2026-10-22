# Model401 inference recovery

Status: recovery for a logger-only interruption; same frozen candidate,
checkpoint set, input lock, inference settings, and one-shot evaluation.

The initial Model401 inference attempt verified its lock, claimed inference,
and saved both seed-mean maps for the first sorted valid image (`A117_step4_sample2`).
It then stopped before the remaining images because a progress print passed
`flush` to `json.dumps`. No valid mask was decoded, no score was calculated,
and no inference manifest was written. Preserve the original Model401 lock,
claim, and two first-image maps.

The recovery lock binds the unchanged Model401 lock/script, the original
inference claim and partial maps, all eight Model392 checkpoints, the B4
archive, exact valid files, and the recovery code. Recovery generates only
missing image maps with the already-locked formula/configuration, refuses to
overwrite existing maps, writes the same complete Model401 inference manifest,
then scores once under the original score claim. It does not rerun the first
image, change any predictions or parameters, tune on valid, or access test.

Use `model401_model397_blend_recovery.py --create-lock`, `--verify-lock`,
`--complete-inference`, then `--score`. Any interruption after the recovery
claim requires a new explicit recovery decision; do not silently rerun.
