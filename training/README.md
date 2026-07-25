# Training on Kaggle

Everything the serving engine needs is trained by one notebook,
[`kaggle_train_ranker.ipynb`](kaggle_train_ranker.ipynb), on a free Kaggle GPU
session (~3 h total, stages checkpoint so a restarted session resumes).

## One-time setup

1. **Create a Kaggle notebook** from `kaggle_train_ranker.ipynb`
   (Kaggle → Code → New Notebook → File → Import Notebook).
2. **Attach a Million Playlist Dataset mirror** (Add Input → search
   "spotify million playlist"). Any dataset containing the raw
   `mpd.slice.*.json` files works — the notebook globs for them.
3. **Create a private dataset** named `music-rec-base-models` with these four
   files and attach it too:

   | file | from |
   |---|---|
   | `item2vec.wordvectors` | `models/item2vec/` |
   | `item2vec.wordvectors.vectors.npy` | `models/item2vec/` |
   | `mood_predictor.pkl` | `models/mood_predictor/` |
   | `features_spec.py` | `training/` (exact copy of the serving feature contract) |

4. Settings → Accelerator → **GPU T4**, then **Run All**.

## What it produces

`artifacts.zip` in the notebook output:

| artifact | purpose |
|---|---|
| `lightgbm_ranker_v2.txt` | LambdaRank ranker over the 22 serving features |
| `ncf_item_v2.pt` | item-NCF (fold-in NeuMF) — the `ncf_score` feature |
| `track_meta.parquet` | real names/artists/durations/popularity for all tracks |
| `metrics.json` | NDCG@10 / Recall@10/50 vs baselines + feature importances |
| `eval_sample.parquet` | 500 held-out test groups for local A/B (`scripts/evaluate.py`) |
| `feature_names.json` | the feature contract the model was trained with |

## Installing the artifacts

Download `artifacts.zip`, then from the repo root:

```bash
python scripts/install_artifacts.py path/to/artifacts.zip
python scripts/evaluate.py          # offline A/B: v2 vs fallback vs seed-cosine
python -m pytest -m slow            # asserts the model matches the serving contract
```

The installer validates the feature contract and requires a passing paired
production release gate before writing any model file. Each accepted file is
streamed into an atomic replacement. The engine picks the new artifacts up on
next start; optional missing capabilities degrade gracefully (see
`services/recommendation/factory.py`).

## Keeping features in sync

`training/features_spec.py` must stay a **byte-identical copy** of
`services/recommendation/features.py` — `tests/test_features.py` enforces this,
and the ranker loader refuses any model whose feature names don't match the
serving contract. If you change a feature: edit `features.py`, re-copy it to
`training/features_spec.py`, update the Kaggle dataset, retrain.

## Learning from feedback

The app stores anonymous impressions and feedback in local SQLite, including
the full candidate slate needed to distinguish displayed, skipped, and unseen
tracks. Export position-adjusted training rows with:

```bash
python scripts/export_feedback.py
```

The export uses inverse propensity weights for display position. Direction
events such as "More similar" remain telemetry rather than positive or
negative labels.
