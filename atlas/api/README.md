# Atlas Backend API (Fase A self-serve)

Vedi `docs/superpowers/specs/2026-07-01-atlas-backend-api-design.md` per il design completo.

## Avvio locale

Richiede Redis in esecuzione (`redis-server` o `docker run -p 6379:6379 redis`).

```bash
cd atlas
export CELERY_BROKER_URL=redis://localhost:6379/0
export CLERK_JWKS_URL=https://your-clerk-instance.clerk.accounts.dev/.well-known/jwks.json
export CLERK_ISSUER=https://your-clerk-instance.clerk.accounts.dev
export STRIPE_API_KEY=sk_test_...

# terminal 1 — API
.venv/bin/uvicorn atlas.api.main:app --reload --port 8000

# terminal 2 — worker (deve girare su Apple Silicon)
.venv/bin/celery -A atlas.api.tasks worker --loglevel=info
```

## Test

```bash
.venv/bin/python -m pytest tests/api/ -v
```
