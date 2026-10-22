# Model391 post-analysis interpretation addendum

This addendum does not change the Model391 preregistered statistic, gate, or
locked source. It records a limitation identified while reviewing the
aggregate result and designing a possible follow-up.

The four native-width strata in the Model391 result contain respectively
16, 15, 21, and 7 specimen groups. These memberships sum to all 59 groups,
with no group represented at multiple widths. Therefore the fitted slope and
its group-bootstrap interval are identified entirely by between-specimen
comparisons. Group bootstrap quantifies sampling uncertainty across specimens
but does not remove confounding between image width and specimen morphology.
The result is an association, not evidence that magnification alone caused
phase-run widths to change.

The unadjusted width-stratum medians also are not uniformly proportional:
between widths 1278 and 1440, run-length medians increase more than native
width for Al, Al3Ni, and Primary Si; Primary Si also reverses between widths
1230 and 1278. These are descriptive strata summaries before the step
adjustment and do not replace the registered slope estimate, but they caution
against treating width as a calibrated physical-scale proxy.

Thus the preregistered gate remains recorded as passed, while its meaning is
narrowed: it permits considering one direct, paired, grouped-OOF experiment
after the already locked Model382/387/388/389 sequence. It neither authorizes
GPU training immediately nor predicts an mIoU gain. Any follow-up must derive
its target width from each outer training fold only and compare against the
same-fold Model382 predictions.
