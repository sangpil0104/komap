# lite15 global model/fold replacement scan

- Baseline: adopted `lite15_N3_R326plain3456`, fresh valid20 image-mIoU 80.6825554175%.
- Scope: the prior verified 294 unique stored-probability candidates plus 8 N3/L2 seed candidates (302 total), rescored against the lite15 baseline; 23 valid fixed replacement/complement rules per candidate. The absent R326 fold1/fold2 slots are excluded; one current R326 slot is evaluated as a single replacement target.
- GPU: none. P100 CPU runner uses 3 processes × 2 Torch threads. No external/user-owned process is stopped.
- Provenance: source candidate probability caches and the stopped lite16 scan remain preserved. Outputs are written to the new P100 experiment folder; no release or checkpoint is replaced. Valid20 is a development screen, not independent generalization evidence. Test GT is not used.
- Before global scoring, the runner checks fresh lite15 R masks, fresh E probabilities and final masks against SHA receipts, then exact valid baseline reconstruction.
