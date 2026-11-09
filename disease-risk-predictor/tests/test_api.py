HIGH = {"Pregnancies": 6, "Glucose": 190, "BloodPressure": 80, "SkinThickness": 35, "Insulin": 200,
        "BMI": 38.0, "DiabetesPedigreeFunction": 1.1, "Age": 55}
LOW = {"Pregnancies": 0, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 20, "Insulin": 70,
       "BMI": 22.0, "DiabetesPedigreeFunction": 0.2, "Age": 24}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_returns_probability(client):
    r = client.post("/predict", json=HIGH)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["risk_level"] in {"low", "moderate", "high"}
    assert body["missing_inputs"] == []


def test_high_profile_scores_above_low_profile(client):
    high = client.post("/predict", json=HIGH).json()["probability"]
    low = client.post("/predict", json=LOW).json()["probability"]
    assert high > low


def test_unknown_measurements_are_accepted(client):
    payload = {**LOW, "Insulin": None, "SkinThickness": None}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert set(r.json()["missing_inputs"]) == {"Insulin", "SkinThickness"}


def test_invalid_values_are_rejected(client):
    assert client.post("/predict", json={**LOW, "Age": 5}).status_code == 422
    assert client.post("/predict", json={**LOW, "Glucose": -10}).status_code == 422
    bad = {k: v for k, v in LOW.items() if k != "Age"}
    assert client.post("/predict", json=bad).status_code == 422


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["test_metrics"]["roc_auc"] > 0.7
    assert len(body["feature_importance"]) == 8


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Diabetes Risk Predictor" in r.text
