# RaceShift API

Local read-only API for model artifacts and dataset metadata.

```bash
pip install -r apps/api/requirements.txt
uvicorn apps.api.main:app --reload --port 8000
```

Training is intentionally not exposed as a public HTTP endpoint. Colab writes model artifacts; the API only reads validated outputs.
