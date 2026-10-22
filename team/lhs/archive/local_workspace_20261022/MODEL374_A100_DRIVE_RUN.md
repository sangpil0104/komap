# Model374 fold 3–4: Colab A100 + Google Drive recovery

The runner uses the already fixed Model374 recipe and only the grouped train
folds 3 and 4. It does not open the official validation or test masks. Training
runs from Colab's local disk; every 5 epochs a verified pair of `last.pt` and
`train_state.pt` is copied to a dedicated Drive folder. The pair includes the
optimizer and RNG state needed for an exact resume. The two newest snapshots
per fold are retained. OOF predictions, summaries, logs, and source payload are
also persisted to Drive.

## Current run (2026-10-01)

The authorized Colab session is `komap374b` on an A100. The job is running in
tmux through the Colab console. Its persistent Drive folder is
`MyDrive/KoMaP/model374_fold34_retry_20261001`. The payload was checked against
SHA-256 `746a10af152d80bd2821262f391ce545f90eb0150b97f4f3fa13043d61c7754e`
before launch. Fold 3 training is in progress; fold 4 follows automatically.
Use `colab status -s komap374b` for session status. Training output is visible
in the existing console connection. A BUSY console status means that connection
is in use; it is not a training failure.

## Recreate on a fresh session if needed

1. Confirm the current session has ended and inspect the newest checksum-verified
   snapshot in Drive. Do not start a second training copy while `komap374b` runs.
2. Create an A100 session from the local terminal:

   ```sh
   colab --auth=oauth2 new --session komap374-recovery --gpu A100
   ```

3. Mount Google Drive for that session using the interactive authorization
   flow. The runner expects it at `/content/drive`:

   ```sh
   colab drivemount -s komap374-recovery
   ```

   Allow roughly 6 GB free in Drive for the two folds' recent checkpoint pairs,
   source/data bundle, and OOF outputs.
4. Upload the fixed payload and two scripts from the local terminal:

   ```sh
   colab upload -s komap374-recovery \
     /Users/hyunseok/orca/workspaces/KoMaP/gurnard/model374_fold34_payload.zip \
     /content/model374_fold34_payload.zip
   colab upload -s komap374-recovery \
     /Users/hyunseok/orca/workspaces/KoMaP/gurnard/run_model374_fold34_colab_a100.sh \
     /content/run_model374_fold34_colab_a100.sh
   colab upload -s komap374-recovery \
     /Users/hyunseok/orca/workspaces/KoMaP/gurnard/experiments/model374_drive_checkpoint_sync.py \
     /content/model374_drive_checkpoint_sync.py
   colab upload -s komap374-recovery \
     /Users/hyunseok/orca/workspaces/KoMaP/gurnard/run_model374_fold34_drive_a100.sh \
     /content/run_model374_fold34_drive_a100.sh
   colab upload -s komap374-recovery \
     /Users/hyunseok/orca/workspaces/KoMaP/gurnard/experiments/model374_train_oof_error_diagnostic.py \
     /content/model374_train_oof_error_diagnostic.py
   ```

5. Launch `bash /content/run_model374_fold34_drive_a100.sh` from a persistent
   console shell in tmux. The runner restores the latest verified checkpoint
   pair from the same Drive folder before continuing. The `colab exec` channel
   has repeatedly lost its WebSocket during long commands in this environment.

   ```sh
   colab console -s komap374-recovery
   ```

The default Drive destination is
`MyDrive/KoMaP/model374_fold34_retry_20261001`. To use a different destination,
set `MODEL374_DRIVE_ROOT` in the Colab runtime before launch. Do not reuse this
folder for a different code or data bundle: the bootstrap refuses to overwrite
different source files.

## Resume after a runtime loss

Create a new A100 session, mount Drive again, upload the same files, then launch
with the same Drive destination. The runner restores the newest checksum- and
weight-verified checkpoint pair and continues from its next epoch. A crash may
lose work since the latest 5-epoch snapshot. A fold is marked complete only
after its grouped-OOF outputs and completion manifest are saved to Drive.

The training-only diagnostic script copied into the immutable source bundle is
`model374_train_oof_error_diagnostic.py`. A separate local follow-up diagnostic,
`experiments/model374_train_oof_oracle_diagnostic.py`, examines class presence
and connected components after all four OOF folds are available. It has not yet
been run on the Colab outputs.
