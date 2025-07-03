# Chevron Anomaly Detection Prototype

This is a prototype solution for detecting anomalies in oil rig operations using sensor data and operator logs. It uses Isolation Forest for anomaly detection, Sentence Transformers for log correlation, and a Hugging Face summarizer for generating GenAI insights.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Key Components

- `generate_data()` – simulates 3 months of hourly sensor data with noise and missing values
- `detect_anomalies()` – uses Isolation Forest to find anomalies
- `generate_logs()` – creates fake operator log entries
- `correlate()` – matches anomalies to logs using semantic similarity
- `generate_insight()` – summarizes each event using a free Hugging Face summarizer
