# Music Recommendation System

A neural music recommendation system with embeddings, learned ranking, and real-time inference.

## Features

- **Neural Embeddings**: MERT and CLMR models for semantic audio understanding
- **Learned Ranking**: XGBoost ranker with collaborative and content features  
- **ANN Search**: FAISS-based approximate nearest neighbor search
- **Diversity Optimization**: MMR and flow-based reordering
- **Real-time API**: FastAPI with <100ms p95 latency
- **Privacy Compliant**: GDPR-ready consent management
- **Production Ready**: Docker, monitoring, feature flags, A/B testing

## Quick Start

```bash
# Install dependencies
make setup

# Start development stack
make dev

# Run the API
make run-api

# Test the endpoints
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Authorization: Bearer dev" \
  -F "audio=@preview.wav" \
  -F "consent={\"terms_version\": \"1.0\"}"

curl "http://localhost:8000/api/v1/recommend?seed=track:123&k=20" \
  -H "Authorization: Bearer dev"
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Embedding      │    │   ANN Index     │
│   - Auth        │───▶│   Pipeline       │───▶│   (FAISS)       │
│   - Validation  │    │   - MERT/CLMR    │    │   - HNSW32      │
│   - Rate Limit  │    │   - Preview Adapt│    │   - Hot Swap    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Consent       │    │   Feature        │    │   Neural        │
│   Manager       │    │   Engineering    │    │   Ranker        │
│   - HMAC Auth   │    │   - Similarity   │    │   - XGBoost     │
│   - Audit Trail │    │   - Popularity   │    │   - Calibrated  │
└─────────────────┘    │   - Graph Dist   │    └─────────────────┘
                       └──────────────────┘             │
                                                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Diversity      │    │   Explanation   │
                       │   - MMR Lambda   │    │   - SHAP Lite   │
                       │   - Flow Optim   │    │   - Heuristics  │
                       └──────────────────┘    └─────────────────┘
```

## Development

### Setup
```bash
make setup          # Install deps and pre-commit hooks
make dev            # Start postgres, redis, minio
```

### Testing
```bash
make test           # Run test suite
make lint           # Check code quality
```

### Building
```bash
make build-index EMB_VER=mert_v2 INDEX_VERSION=ann_v1
make serve-ann INDEX_VERSION=ann_v1
make eval INDEX_VERSION=ann_v1
```

## Configuration

Configuration is managed through YAML files with environment overrides:

- `conf/default.yaml` - Base configuration
- `conf/feature_flags.yaml` - Feature flags and A/B tests
- Environment variables override config values

## API Endpoints

### POST /api/v1/analyze
Analyze audio preview and generate embeddings.

**Request:**
```bash
curl -X POST "/api/v1/analyze" \
  -H "Authorization: Bearer <jwt>" \
  -H "X-Request-ID: <uuid>" \
  -F "audio=@preview.wav" \
  -F "consent={\"terms_version\": \"1.0\", \"retention_days\": 90}"
```

**Response:**
```json
{
  "canonical_uid": "isrc:USUM71703861",
  "embedding": {
    "model": "mert",
    "version": "v2",
    "dimension": 256,
    "vector_id": "emb_USUM71703861"
  },
  "features": {
    "tempo": 128.5,
    "key_signature": "C",
    "energy": 0.847,
    "valence": 0.621
  },
  "provenance": {
    "models_used": ["mert_v2"],
    "processing_time_ms": 45,
    "consent_id": "consent_abc123"
  }
}
```

### GET /api/v1/recommend
Get personalized music recommendations.

**Request:**
```bash
curl "/api/v1/recommend?seed=isrc:USUM71703861&k=20&diversity_lambda=0.7" \
  -H "Authorization: Bearer <jwt>"
```

**Response:**
```json
{
  "seed": "isrc:USUM71703861",
  "recommendations": [
    {
      "canonical_uid": "isrc:GBUM71505078",
      "title": "Blinding Lights", 
      "artist": "The Weeknd",
      "similarity_score": 0.94,
      "rank_score": 0.87,
      "features": {"tempo": 171, "energy": 0.73},
      "explanation": {
        "top_factors": ["Similar tempo", "Same key signature"],
        "similarity_reason": "High cosine similarity in embedding space"
      }
    }
  ],
  "pipeline": {
    "embedding_model": "mert_v2",
    "index_version": "ann_v1",
    "ranker_version": "xgb_v1"
  },
  "timing_ms": {
    "ann_search": 8,
    "ranking": 12,
    "mmr_flow": 5
  }
}
```

## Latency Budgets

Target p95 latency per component:
- Consent validation: ≤2ms
- Identity resolution: ≤3ms  
- Embedding lookup: ≤5ms (cache) / ≤35ms (compute)
- ANN search: ≤8ms
- Feature engineering: ≤10ms
- Neural ranking: ≤10ms
- MMR + flow: ≤5ms
- **Total: ≤100ms (warm)**

## Privacy & Compliance

- **Consent Management**: HMAC-based user/IP hashing
- **Preview Enforcement**: ≤30 second audio clips only
- **Audit Trail**: Immutable consent and processing logs
- **Data Retention**: Configurable TTL with automatic cleanup
- **GDPR Ready**: Right to deletion, data portability

## Deployment

```bash
# Build and deploy
docker compose build
docker compose up -d

# Check health
curl http://localhost:8000/health

# Monitor metrics
curl http://localhost:8000/metrics
```

## Monitoring

- **Structured Logs**: JSON format with request tracing
- **Prometheus Metrics**: Latency, throughput, error rates
- **Health Checks**: Database, Redis, S3 connectivity
- **Performance Budgets**: Automatic alerting on latency violations

## License

MIT License - see LICENSE file for details.