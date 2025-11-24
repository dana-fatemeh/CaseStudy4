# Case Study 4: Industrial Equipment Anomaly Detection

## Objective
Build an anomaly detection system that predicts equipment failures before they happen.

---

## Dataset
**[Controlled Anomalies Time Series (CATS)](https://huggingface.co/datasets/patrickfleith/controlled-anomalies-time-series-dataset)** (Hugging Face)

4M observations, 18-24 sensors, 200 labeled anomalies with root cause metadata

---

## Business Problem
Manufacturing plant: 20-30% unplanned equipment failures costing $100K-$500K each. Goal: Predict failures 24-48 hours in advance.

---

## Methods

### Basic ML
- Isolation Forest (anomaly detection)
- Random Forest (classification)
- XGBoost (root cause)
- Statistical methods (Z-score, IQR)

**Accuracy:** 78-85% | **Speed:** Fast | **Interpretability:** High

### Advanced ML
- Autoencoder (deep learning)
- LSTM/GRU (temporal patterns)
- One-Class SVM
- Variational Autoencoder

**Accuracy:** 85-93% | **Speed:** Slower | **Interpretability:** Low

---

## Product: Monitoring API

```
POST /api/v1/detect-anomaly
{
  "sensor_data": [22.5, 101.3, 45.2, 18.9, ...],
  "timestamp": "2025-11-12T14:30:00Z"
}

Response:
{
  "anomaly_detected": true,
  "anomaly_score": 0.87,
  "anomaly_type": "thermal_failure",
  "root_cause_sensor": "temperature_sensor_3",
  "time_to_failure_hours": 36
}
```

**Stack:** FastAPI | TensorFlow | scikit-learn | Streamlit | Docker

---

## Business Value
- Prevent failures: $200K-$1.5M savings per client
- SaaS model: $5K-$50K/month per plant

---

## Libraries
- [scikit-learn](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [SHAP](https://github.com/slundberg/shap)