# Legacy experiment launchers

These four launchers were present in the experiment workspace but missing from the first GitHub archive. They are retained for provenance and to complete the source inventory. They are not part of the current Model336–348 queue.

| File | Historical purpose | Re-run caveat |
|---|---|---|
| `run_stronger_sequence.sh` | Sequential backbone trials associated with Models 82–90 | Waits for historical `/proc/435828`; selects GPU 3. Update the process guard and device before reuse. |
| `run_effv2_recovery.sh` | EfficientNetV2-S recovery trial (Model 84) | Waits for historical `/proc/436112`; selects GPU 3. Update the process guard and device before reuse. |
| `queue_model326_after325.sh` | Queue Model326 after Model322 and Model325 complete | Uses the old output layout and GPU 6 headroom check; verify paths and prerequisites before reuse. |
| `run_model326_metric_aligned_group_oof.sh` | Earlier metric-aligned EfficientNet-B4 grouped-OOF variant | Expensive four-fold, 16,000-update recipe; distinct from the later selected Model326 full-image-stat pipeline. |

Competition data, checkpoints, and generated probability maps are not included. These scripts may need code/path adaptation and should not be launched blindly.
