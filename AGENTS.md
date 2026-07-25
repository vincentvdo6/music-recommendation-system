# One-Rec — Music Recommendation Engine

Hybrid recommender serving 597K tracks: item2vec retrieval + LightGBM LambdaRank
ranking. Public repo (resume piece): github.com/vincentvdo6/music-recommendation-system.

## Architecture (one pipeline, no dead paths)

```
static/ (vanilla ES modules, strict CSP)
  → api/main.py → api/routers/search.py
  → services/music/service.py            (orchestration + Spotify enrichment)
  → services/recommendation/engine.py    (THE pipeline)
      retrieve (70% seed ANN / 30% playlist mean + exact acoustic neighbors)
      → features.build_matrix (22 features, vectorized)
      → shared LambdaRank model (linear fallback)
      → acoustic discovery reservation → deep enrichment → artist/era diversity
      → top-N + SHAP-style explanations + anonymous impression feedback
```

## Key modules

- `services/recommendation/features.py` — `FEATURE_NAMES` contract; **byte-identical
  mirror at `training/features_spec.py`** (enforced by test). Change one → re-copy the other.
- `services/recommendation/factory.py` — loads artifacts with per-artifact graceful
  degradation; logs a capability table at startup. `get_engine()` singleton.
- `services/recommendation/ncf.py` — item-NCF fold-in scorer; torch imported lazily
  (app runs without torch).
- `services/recommendation/track_meta.py` — MPD-derived artists/popularity/durations
  (`models/meta/track_meta.parquet`).
- `services/music/service.py` — playlist resolution, Spotify metadata enrichment
  (drops tracks Spotify doesn't know), artist dedup + children's-music blacklist.
- `training/kaggle_train_ranker.ipynb` — full training pipeline on Kaggle GPU
  (see `training/README.md`). Artifacts installed via `scripts/install_artifacts.py`.

## Hard constraints

- **Spotify audio-features + recommendations APIs are deprecated** (403/404).
  Never reintroduce them. Mood comes from the embedding-based predictor.
- **numpy<2 pin is load-bearing** (gensim/annoy ABI) — keep it in requirements,
  requirements-ci, and the notebook.
- The ranker loader asserts `model.feature_name() == FEATURE_NAMES` and refuses
  mismatches — retrain rather than bypass.
- MPD data ends in 2017: newer seeds resolve via same-artist proxy (needs track_meta).
- Git commits: NO Co-Authored-By trailers (resume repo).

## Commands

```bash
python run_local.py                        # dev server :8000 (system Python 3.12 has all deps)
python -m pytest -m "not slow" -q          # fast suite (fake model stack) — CI runs this
python -m pytest -m slow -q                # real artifacts, feature-contract assertions
python scripts/evaluate.py                 # offline A/B on evaluation/eval_sample.parquet
python scripts/install_artifacts.py X.zip  # install Kaggle training output
python scripts/export_feedback.py          # local SQLite → IPS-weighted parquet
python scripts/report_feedback.py          # summarize live anonymous outcomes
python scripts/download_models.py          # fetch model weights from GitHub release
```

Windows: use `C:\Users\Vincent\AppData\Local\Programs\Python\Python312\python.exe`
(the `.venv` is empty; system Python has the full stack including torch).

## Models on disk (`models/`, gitignored, ~1.4 GB serving set)

- `item2vec/item2vec.wordvectors(+.vectors.npy)` — 597,682 × 200-dim
- `item2vec/ann_index/{annoy.index,id_map.json}`
- `ranker/lightgbm_ranker_v2.txt` — LambdaRank (v2 feature contract)
- `ncf/ncf_item_v2.pt` — item-NCF checkpoint (`format: item-ncf-v2`)
- `meta/track_meta.parquet` — MPD metadata for all tracks
- `mood_predictor/mood_predictor.pkl`
- `policy.json` — validation-frozen shared serving policy
- `acoustic_policy.json` — optional acoustic discovery-lane settings

Missing artifacts degrade gracefully (fallback ranker, zeroed features) — check
the "Engine capabilities" startup log line when debugging.
