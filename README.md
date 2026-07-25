# One-Rec

**Paste a playlist. Search a song. Get one great recommendation at a time.**

A hybrid music recommendation engine serving 597K tracks — item2vec embeddings
trained on the Spotify Million Playlist Dataset, ANN retrieval, and a LightGBM
LambdaRank model over signals that actually exist at serving time.

![One-Rec UI](docs/ui-light.png)

<details>
<summary>Dark mode</summary>

![One-Rec dark mode](docs/ui-dark.png)

</details>

## How it works

```mermaid
flowchart LR
    UI["One-Rec UI<br/>(vanilla JS, no framework)"] --> API["FastAPI"]
    API --> ENGINE["Recommendation engine"]

    subgraph ENGINE_DETAIL ["services/recommendation"]
        RETRIEVE["candidate retrieval<br/>seed + playlist mean (item2vec)<br/>+ exact acoustic neighbors"]
        FEATURES["22-feature matrix<br/>(vectorized, numpy)"]
        RANK["LightGBM LambdaRank<br/>(linear fallback)"]
        DIVERSITY["acoustic discovery slots<br/>+ artist / era diversity"]
        RETRIEVE --> FEATURES --> RANK --> DIVERSITY
    end

    ENGINE --> ENGINE_DETAIL
    NCF["item-NCF<br/>(fold-in NeuMF)"] --> FEATURES
    MOOD["mood predictor<br/>(from embeddings)"] --> FEATURES
    META["track_meta.parquet<br/>(MPD popularity/artists)"] --> FEATURES
    DIVERSITY --> SPOTIFY["Spotify metadata<br/>enrichment"]
```

1. **Retrieval** — the searched song ("seed") drives 70% of the collaborative
   pool via its Annoy neighbors; the mean of the remaining playlist supplies
   the rest. Exact acoustic neighbors add sound-alikes from the preview
   embedding catalog, and newer seeds can use a same-artist collaborative
   proxy when they fall outside the 2017 playlist vocabulary.
2. **Ranking** — every candidate gets a 22-feature vector (seed/playlist
   cosines, ANN reciprocal ranks, item-NCF score, MPD popularity prior, artist
   overlap, mood similarity, duration fit, acoustic similarity and channel
   availability) scored by a LambdaRank model trained on leave-N-out playlist
   continuation with **the exact serving retrieval**.
3. **Serving** — an independent acoustic lane reserves a few discovery
   slots, then deep over-fetching, Spotify identity resolution, one-per-artist
   and a soft era cap produce a full playable list. Apple fills missing media.

Every recommendation ships its explanation — the top SHAP-style feature
contributions — and the similarity shown in the UI is the real seed cosine,
hidden when the seed isn't in the model rather than faked.

### Honest-signals design

Spotify deprecated its audio-features and recommendations APIs (403/404 for
new apps), which silently killed most classic feature sets. Everything here is
computable offline at serving time:

| Signal | Source |
|---|---|
| Co-listen similarity | item2vec (200-dim, trained on 1M playlists) |
| **Acoustic similarity** | Discogs-EffNet embeddings of 30s preview clips (46K tracks, PCA-256) — what songs actually *sound* like |
| Collaborative score | item-NCF (fold-in NeuMF, BPR loss) — scores unseen playlists by mean-pooling |
| Mood (valence/energy/…) | regression from item2vec embeddings, trained while the audio API existed |
| Popularity prior, artists, durations | aggregated from the Million Playlist Dataset |

A byte-identical feature specification is shared between serving and the
training notebook. The model loader refuses feature-name mismatches so
training/serving drift fails safely.

## Offline evaluation

Metrics are **end-to-end and unconditional**: queries whose positives never
survive retrieval count as zero-scoring failures instead of being silently
dropped (the usual way recommender metrics flatter themselves). Held-out
playlists, leave-N-out, candidates from the exact serving retrieval path,
scored through the exact serving policy — from `models/metrics.json`,
survivor-conditional slice reproduced locally by
[`scripts/evaluate.py`](scripts/evaluate.py):

| System (true end-to-end, test) | NDCG@10 | Recall@10 | Recall@50 | Hit@1 | MRR |
|---|---|---|---|---|---|
| **This release — with playlist context** | **0.156** | **0.110** | **0.209** | **0.229** | **0.332** |
| **This release — seed-only** | **0.184** | **0.034** | **0.089** | **0.286** | **0.393** |
| previous release, exact-served — with playlist | 0.108 | 0.075 | 0.131 | 0.161 | 0.246 |
| previous release, exact-served — seed-only | 0.154 | 0.029 | 0.069 | 0.217 | 0.327 |

The published production upgrade is gated on a paired bootstrap against the previous system served
exactly as it ran: **+0.039 NDCG@10, 95% CI [+0.036, +0.043]** — the win comes
from a wider, uncapped candidate funnel, not from grading on a curve. The
single top recommendation is a held-out playlist track 23–29% of the time
(Hit@1 matters here — the UI promises *one* great recommendation). Seed-only
requests (no playlist context) are their own slice with their own funnel.

The serving policy is frozen on validation, not hand-tuned: a grid over the
`SEED_AFFINITY` and `DISCOVERY` dials, constrained to keep seed similarity at
parity with the previous system and popularity below its served level, trades
~1 point of raw NDCG@10 (0.168 → 0.156) for recommendations that stay close to
the seed's sound and lean discovery. Ablations in `metrics.json` are honest:
the audio and mood features are ranking-neutral on this test set (their value
is audible, not measurable in NDCG), and a per-mode stagewise funnel
attributes every lost positive to retrieval or a specific filter.

New candidates must carry a completed paired release gate. The installer reads
that decision before writing any model file, so a survivor-conditional local
win cannot replace production after losing the true end-to-end comparison.

## Running it

```bash
pip install -r requirements.txt
python scripts/download_models.py     # ~1.4 GB from the GitHub release
cp .env.example .env                  # add your Spotify client credentials
python run_local.py                   # http://localhost:8000
```

If port 8000 is occupied, PowerShell can launch another verified instance with
`$env:PORT=8001; python run_local.py`.

Or with Docker (models stay volume-mounted, never baked into the image):

```bash
docker compose up --build
```

Spotify credentials come from a free [developer app](https://developer.spotify.com/dashboard)
and are only used for search + display metadata.

The seed page exposes three serving profiles: **Closest match** (the default),
**Balanced**, and **Explorer**. They are bounded post-ranking policies, so changing profile does
not change the trained feature or retrieval contracts. Likes, dislikes,
playback events, “More like this,” and “More similar” tune subsequent results
inside the current anonymous browser session. Feedback influence decays with a
24-hour half-life, and tracks shown during the last day are rotated out of
refreshed slates so tuning does not simply return the same list. Preview skips,
completions, and replays are distinguished automatically.
Neutral feedback is logged for analysis but applies no positive or negative
session adjustment and is excluded from training labels.

## Training

The full pipeline — MPD parsing, item-NCF training, ranker dataset
construction, LambdaRank training, evaluation — runs in one Kaggle notebook on
a free GPU session (~3h). See [training/README.md](training/README.md).

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/search` | track search (Spotify, Apple fallback) |
| `POST` | `/api/v1/playlist/import` | resolve a public playlist URL to entries |
| `POST` | `/api/v1/playlist/recommendations` | seed-driven recommendations |
| `POST` | `/api/v1/feedback` | local, anonymous impression feedback |
| `GET` | `/health` | health check |

## Tests

```bash
pytest -m "not slow"    # fast suite, fake model stack (runs in CI)
pytest -m slow          # loads the real 2.3 GB artifacts, asserts the feature contract
python scripts/evaluate_profiles.py  # compare the three serving profiles offline
python scripts/report_feedback.py    # compare live profile/session outcomes
python scripts/export_feedback.py    # export IPS-weighted candidate feedback
```

## Stack

FastAPI · gensim (item2vec) · Annoy · LightGBM · PyTorch (optional, lazy) ·
pandas/pyarrow · vanilla ES-module frontend with a strict CSP (no frameworks,
no inline scripts, self-hosted fonts).
