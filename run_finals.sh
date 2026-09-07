#!/usr/bin/env bash
# Build the two final submissions on the signup line.
#
# Everything here is deterministic and pinned:
#   * `--deterministic` fixes the thread count for all three families, so the
#     fits are repeatable given the same matrix (rulebook 8.2).
#   * `--rounds-book tune_rounds.s5_asbuilt.json` pins the round counts that
#     actually produced 0.55014 on the board. Today's re-run of tune.py moved
#     `lgb_deep` from 230 to 758 rounds for a tail_late change of 0.0011, so
#     "the same members" is not enough to identify the same model.
#   * The 244-column matrix is copied aside afterwards, because the GNN tier is
#     not bit-reproducible across builds (max abs diff 3e-4, correlation
#     1.000000) and that is enough to move an early stop by 3x. Shipping the
#     matrix is what makes the notebook reproduce exactly rather than closely.
#
# s6 and s7 differ only in weighting, and submit.py caches each member's
# seed-averaged test ranking, so s7 costs seconds rather than a second refit.
set -euo pipefail
cd /g/REACT_datathon
PY=.venv/Scripts/python.exe
BOOK=tune_rounds.s5_asbuilt.json

# The integrity feature is enabled for this build only, through the environment.
# The first version of this script rewrote the literal in src/config.py and
# restored it at the end, which left the repo on the wrong value twice when the
# process was killed before the restore ran. An exported variable cannot outlive
# the shell that set it.
export REACT_SIGNUP=1
$PY -c "from src import config as C, harness as H; \
print('signup:', C.USE_SIGNUP_INCONSISTENCY, '| feature hash:', H.feature_source_hash())"

# Member-equal over the five hyperparameter configs plus CatBoost: submission
# 5's composition exactly, which is the best measured on the board (0.55014).
$PY -u submit.py --name s6_det_s5comp --seeds 3 --deterministic \
  --rounds-book "$BOOK" \
  --weights '{"lgb_base":1,"lgb_deep":1,"lgb_shallow":1,"lgb_reg":1,"lgb_extra":1,"cat_base":1}' \
  --note "Submission 5's composition rebuilt deterministically at 3 seeds, pinned to the as-built round book. Spent on reproducibility, not score: s5 was fitted with deterministic=False at 2 seeds and its feature matrix has since been overwritten, so it cannot be reproduced from a notebook and rulebook 8.2 forfeits a position that cannot be. Expected board delta ~0; a large move would instead be evidence that the run-to-run floor is wider than 0.003."
echo "=== S6 DONE ==="

echo "=== preserving the 244-column matrix that produced these fits ==="
cp data/processed/base_features.parquet data/processed/base_features.signup.parquet
cp data/processed/base_features.stamp.json data/processed/base_features.signup.stamp.json
ls -la data/processed/base_features.signup.parquet

# Same six members, weighted one-half CatBoost / one-half the LightGBM core --
# equal weight per *family* rather than per member. A-priori (nothing is fitted
# to the tail) and it measured +0.0007 on tail_late with a paired interval of
# [-0.0003, +0.0018]: inside the noise floor, which is why it is a second
# candidate rather than an upgrade.
$PY -u submit.py --name s7_family_equal --seeds 3 --deterministic \
  --rounds-book "$BOOK" \
  --weights '{"lgb_base":0.1,"lgb_deep":0.1,"lgb_shallow":0.1,"lgb_reg":0.1,"lgb_extra":0.1,"cat_base":0.5}' \
  --note "Same six members as s6, weighted equally per family (1/2 CatBoost, 1/2 spread over the five LightGBM configs) instead of equally per member. Local tail_late +0.0007 [-0.0003, +0.0018], mean(late,far) +0.0007 -- below the floor, so this is spent to have a second scored candidate at equal expected score rather than to climb."
echo "=== S7 DONE ==="

# Nothing to restore: REACT_SIGNUP was exported into this shell only, and
# src/config.py was never touched.
$PY -c "from src import config as C; print('repo default is still', C.USE_SIGNUP_INCONSISTENCY)"
echo "=== ALL DONE ==="
