from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
from fastapi.testclient import TestClient
from backend.api import app
from core.schema import FEATURE_COLUMNS

client = TestClient(app)
assert client.get('/api/health').status_code == 200
row = pd.read_csv('data/sample_mixed_traffic.csv').iloc[[0]]
features = json.loads(row[FEATURE_COLUMNS].to_json(orient='records'))[0]
r = client.post('/api/predict', json={'features': features, 'telemetry': {}})
assert r.status_code == 200, r.text
assert 'current_stage' in r.json()
assert 'forecast' in r.json()
print('ChronoFlux smoke test: PASS')
