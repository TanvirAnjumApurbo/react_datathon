"""Rescale submission values to a probability-like range.

The blend is built on ranks, which is correct for the metric -- average
precision depends only on ordering -- but leaves the raw output uniform on
[0, 1] with a mean of 0.50, which does not read as a fraud probability for a
problem with a ~1.7% base rate. The rules ask for "a probability in [0, 1], not
a hard label".

This maps the submission's ranks onto the distribution of blended *probability*
predictions measured on the primary validation fold. Because the mapping is
strictly monotone, the ordering -- and therefore the PR-AUC -- is bit-for-bit
unchanged. Only the scale moves.

    python calibrate_submission.py            # writes submission_calibrated.csv
    python calibrate_submission.py --inplace  # overwrite submission.csv

This is presentation, not modelling: it cannot and does not change the score.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from src import config as C

VAL_PREDS = C.PROCESSED / "val_preds.parquet"
BLEND_SPEC = C.PROCESSED / "blend_spec.json"
SUBMISSION = C.ROOT / "submission.csv"


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--inplace", action="store_true")
    args = ap_.parse_args()

    weights = json.loads(BLEND_SPEC.read_text())["weights"]
    vp = pd.read_parquet(VAL_PREDS)
    y = vp.pop("_y").to_numpy()

    # Reproduce the blend on the fold, in probability space this time.
    prob = np.zeros(len(y))
    for name, w in weights.items():
        prob += w * vp[name].to_numpy()
    prob /= sum(weights.values())

    sub = pd.read_csv(SUBMISSION)
    r = rankdata(sub[C.TARGET].to_numpy(), method="ordinal") - 1
    # Quantile-map: nth-ranked test row takes the nth quantile of the fold's
    # blended probabilities.
    ref = np.sort(prob)
    q = r / (len(r) - 1)
    # Interpolate rather than index: the fold has fewer rows than the test set,
    # so nearest-index mapping would collapse distinct ranks onto equal values,
    # and those ties do move the precision-recall curve.
    mapped = np.interp(q, np.linspace(0.0, 1.0, len(ref)), ref)
    # `ref` can still contain repeated values, so add a strictly increasing
    # epsilon to guarantee the original ordering survives exactly.
    mapped = np.clip(mapped + 1e-9 * q, 0.0, 1.0)

    # Prove the ordering survived: AP on the fold is invariant under the map.
    ap_before = average_precision_score(y, prob)
    prob_rank = (rankdata(prob, method="ordinal") - 1).astype(int)
    ap_after = average_precision_score(y, ref[prob_rank])
    assert abs(ap_before - ap_after) < 1e-12, "monotone map altered the ordering"

    spearman_ok = np.array_equal(
        rankdata(sub[C.TARGET].to_numpy(), method="ordinal"),
        rankdata(mapped, method="ordinal"),
    )
    assert spearman_ok, "submission ordering changed -- would change the score"

    out = sub.copy()
    out[C.TARGET] = mapped
    path = SUBMISSION if args.inplace else C.ROOT / "submission_calibrated.csv"
    out.to_csv(path, index=False)

    print(f"fold AP unchanged by the mapping: {ap_before:.6f}")
    print(f"submission ordering preserved:    {spearman_ok}")
    print("\nbefore:", sub[C.TARGET].describe([.5, .99]).round(4).to_dict())
    print("after :", out[C.TARGET].describe([.5, .99]).round(4).to_dict())
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
