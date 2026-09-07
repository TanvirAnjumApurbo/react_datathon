"""Which local statistic actually predicts the leaderboard?

    .venv/Scripts/python.exe -u horizon_proxy.py

`tail_late` is the repo's selection target, and four submissions have measured
exactly how far that trust extends: it passes large gains through at 29-41%
(+0.035 local -> +0.0144 board; +0.025 -> +0.0072) and **loses the sign** below
roughly 0.008 -- submission 3 gained +0.0023 on `tail_late` and lost 0.0025 on
the board. Everything Stage 2 measured lives under that line, which is why the
remaining submission budget is hard to spend: the instrument cannot resolve the
candidates that are left.

This script tests a replacement. Submission 2's board score (0.54151) landed
between its `tail_late` (0.5507) and `tail_far` (0.5304) and close to their mean
(0.5406) -- which is what you would expect geometrically, since the real test
window spans 1-62 days past its cutoff and then keeps drifting for another two
months of calendar time, while `tail_late` reads 46-62 days and `tail_far` reads
75-91. One coincidence is not a finding. Two submissions on the same feature
line, scored the same way, is a test.

The test is deliberately the *hard* case. On the two clean-line submissions:

    board:      s2 (equal_core) 0.54151  >  s3 (equal_lgb_all) 0.53901
    tail_late:  s2 0.5507                <  s3 0.5530            <- wrong sign

So `mean(tail_late, tail_far)` earns the job only if it recovers the board's
ordering on a comparison `tail_late` gets backwards. Anything less is a
statistic that agrees where agreement was never in doubt.

Nothing here fits a model. It reads the predictions `tune.py --far` persisted
and rank-averages a-priori compositions -- the same unweighted groupings
`blend.py` is allowed to adopt outright, because none of them learns a weight
from the evaluation window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import config as C
from src.validation import ap
from tune import PARAM_CANDS

TAIL_PREDS = C.PROCESSED / "tune_tail_preds.parquet"
HORIZONS = ("recent", "late", "far")

# Submission 2's five hyperparameter configs, taken from the candidate list
# rather than matched by an `lgb` prefix. The diverse members added later are
# also LightGBM, so a prefix would quietly redefine the one composition whose
# board score this script exists to check against.
CORE = [c.name for c in PARAM_CANDS]

# Public leaderboard results for compositions that were actually uploaded, so a
# proposed proxy can be scored against reality rather than against taste.
#
# Only clean-line (243-column) entries appear. s4 and s5 were built on the
# 244-column signup matrix, so their board scores are not comparable with local
# numbers computed here, and including them would quietly compare two different
# feature sets and call the difference proxy error.
BOARD = {
    "equal_core": (0.54151, "s2_pipeline_equal_lgb"),
    "equal_lgb_all": (0.53901, "s3_equal_lgb_all"),
}


def load() -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """-> {member: {horizon: rank vector}}, y."""
    if not TAIL_PREDS.exists():
        raise SystemExit("run tune.py --far first")
    vp = pd.read_parquet(TAIL_PREDS)
    y = vp.pop("_y").to_numpy()
    n = len(y)
    members: dict[str, dict[str, np.ndarray]] = {}
    for col in vp.columns:
        name, _, tag = col.partition("@")
        members.setdefault(name, {})[tag] = rankdata(vp[col].to_numpy()) / n
    return members, y


def compositions(members: dict) -> dict[str, list[str]]:
    """The a-priori groupings, named for the submission that shipped them.

    Membership is decided by what a member *is*, never by what it scored on the
    window being used to judge it. That distinction is the whole reason these
    are adoptable on a point estimate while a hill-climb is not.
    """
    core = [m for m in CORE if m in members]
    lgb_all = core + [m for m in members if m.startswith(("hl_", "win_"))]
    out = {
        "equal_core": core,                       # submission 2
        "equal_lgb_all": lgb_all,                 # submission 3
        "core_plus_cat": core + ["cat_base"],     # submission 5's shape, clean line
        "equal_all": list(members),
    }
    return {k: v for k, v in out.items() if v and all(m in members for m in v)}


def score_group(members: dict, names: list[str], y: np.ndarray) -> dict:
    """AP of the equal-weight rank average at every horizon it can be read at.

    A composition is scored at a horizon only where *every* member has a
    prediction. Averaging nine members at `far` against five at `late` would
    make the two horizons different models, and the whole point of the
    comparison is that they are the same model read at different distances.
    """
    row: dict[str, float] = {}
    for h in HORIZONS:
        if not all(h in members[m] for m in names):
            row[h] = np.nan
            continue
        row[h] = ap(y, np.mean([members[m][h] for m in names], axis=0))
    row["mean_late_far"] = 0.5 * (row["late"] + row["far"])
    row["decay"] = row["late"] - row["far"]
    return row


def main() -> None:
    members, y = load()
    n_far = sum("far" in v for v in members.values())
    print(f"\n{len(members)} members, {len(y):,} tail rows, base rate {y.mean():.4f}")
    print(f"far coverage: {n_far}/{len(members)} members")
    if n_far < len(members):
        missing = sorted(m for m, v in members.items() if "far" not in v)
        print(f"  MISSING far: {missing}")
        print("  -> a composition containing these cannot be scored at far, and "
              "comparing one\n     that can against one that cannot is confounded "
              "rather than informative.")

    rows = []
    for m, v in sorted(members.items()):
        r: dict = {"member": m}
        for h in HORIZONS:
            r[h] = ap(y, v[h]) if h in v else np.nan
        r["mean_late_far"] = 0.5 * (r["late"] + r["far"])
        r["decay"] = r["late"] - r["far"]
        rows.append(r)
    mem = pd.DataFrame(rows).sort_values("mean_late_far", ascending=False)
    print("\n=== MEMBERS (recent = 1-17d ahead, late = 46-62d, far = 75-91d) ===")
    print(mem.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n  `decay` is late - far: what a member loses when the drift gets "
          "another month to\n  work. The real test window ends 62 days past its "
          "cutoff and then keeps drifting,\n  so a flat decay is worth more than "
          "the point estimate alone suggests.")

    comp = compositions(members)
    rows = []
    for label, names in comp.items():
        r = {"composition": label, "n": len(names), **score_group(members, names, y)}
        board, fname = BOARD.get(label, (np.nan, ""))
        r["board"] = board
        r["file"] = fname
        rows.append(r)
    cf = pd.DataFrame(rows)
    cf["err_tail_late"] = cf["late"] - cf["board"]
    cf["err_mean"] = cf["mean_late_far"] - cf["board"]
    print("\n=== COMPOSITIONS ===")
    print(cf[["composition", "n", "recent", "late", "far", "mean_late_far",
              "decay", "board", "err_tail_late", "err_mean"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== THE TEST ===")
    have = cf[cf["board"].notna() & cf["mean_late_far"].notna()]
    if len(have) < 2:
        print("  not enough board-scored compositions with full far coverage; "
              "fill the far reads and re-run.")
        return
    a, b = have.iloc[0], have.iloc[1]
    d_board = a["board"] - b["board"]
    d_late = a["late"] - b["late"]
    d_mean = a["mean_late_far"] - b["mean_late_far"]
    late_ok = np.sign(d_late) == np.sign(d_board)
    mean_ok = np.sign(d_mean) == np.sign(d_board)
    print(f"  {a['composition']} vs {b['composition']}:")
    print(f"    board            {d_board:+.5f}")
    print(f"    tail_late        {d_late:+.5f}   {'AGREES' if late_ok else 'WRONG SIGN'}")
    print(f"    mean(late,far)   {d_mean:+.5f}   {'AGREES' if mean_ok else 'WRONG SIGN'}")
    print()
    if mean_ok and not late_ok:
        print("  PROMOTE mean(tail_late, tail_far) to the selection objective, and "
              "hold it loosely.\n"
              f"  It recovers an ordering tail_late gets backwards -- but the board "
              f"delta being\n  reproduced is {abs(d_board):.4f}, which is at the "
              "~0.003 board noise floor, and there is\n  exactly one like-for-like "
              "comparison to be had. That makes this suggestive, not\n  established: "
              "enough to prefer it as a tie-breaker between candidates tail_late\n"
              "  cannot separate, not enough to spend a submission on a delta it "
              "alone reports.")
    elif mean_ok:
        print("  Both statistics agree with the board here, so this comparison does "
              "not separate\n  them. No reason to switch; keep tail_late and its "
              "0.008 resolution limit.")
    else:
        print("  REJECT. mean(tail_late, tail_far) does not recover the board's "
              "ordering either.\n  The 0.008 limit stands: admit members on the "
              "decorrelation gate (rank rho <= 0.70\n  against the blend, which has "
              "direct board evidence behind it) and do not spend a\n  submission on "
              "a local delta below it.")
    print("\n  Caveat that applies whichever way this lands: the two board scores "
          "differ by\n  0.0025, which is inside the run-to-run floor for the "
          "leaderboard itself. One\n  comparison at the edge of the noise cannot "
          "establish a proxy; it can only fail to\n  refute one. Read the `decay` "
          "column and the decorrelation gate as the primary\n  evidence, and this "
          "as the tie-breaker it is.")


if __name__ == "__main__":
    main()
