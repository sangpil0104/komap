# R ensemble inference dependency.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


def route_params(mean, std):
    if 159.0 <= mean < 161.0:
        return {"high_weight": .70, "pair_weight": .05, "style_weight": 1.0,
                "al_bias": .45, "beta": 10.0, "eut_bias": -1.5}
    return {"high_weight": .85 if mean < 165.0 else .55,
            "pair_weight": .10, "style_weight": .75 if std >= 50 else 0.0,
            "al_bias": .45,
            "beta": (15.0 if 164.0 <= mean < 164.5 or 174.6 <= mean < 174.85
                     else (3.0 if mean < 160 else 7.0)),
            "eut_bias": (-2.0 if 161.3 <= mean < 162.0 else
                         (-.75 if 164.5 <= mean < 165.2 else
                          (.3 if 174.0 <= mean < 175.5 else (-1.0 if mean < 165 else -.3))))}
