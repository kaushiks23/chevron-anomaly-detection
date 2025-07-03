# Chevron Anomaly Detection Report

## Pipeline Architecture

1. **Data Simulation**:
   - 3 months of hourly data
   - Noise added using `tsaug`
   - 5% missing values injected randomly

2. **Anomaly Detection**:
   - Isolation Forest (unsupervised)
   - Contamination set to 5%

3. **Log Generation**:
   - Sampled every 50 rows with placeholder log text

4. **Correlation**:
   - Sentence-BERT used to encode both anomalies and logs
   - Cosine similarity used to match

5. **GenAI Insight**:
   - Prompt-based summarization using Hugging Face's DistilBART

## Key Decisions & Trade-offs

- Used semantic similarity instead of just timestamp matching
- Selected lightweight models (free-tier friendly)
- Streamlit app for fast prototyping over Flask

## Failure Points

- No ground-truth to evaluate anomaly detection
- If anomaly is repetitive, isolation forest may miss it
- If there is a gradual pressure increase over time, isolation forest may miss it
- Using only limited features
- No ground-truth to evaluate anomaly detection
- Operator logs are synthetic and repetitive
- Correlation assumes log text is meaningful

## Future Work

- Train on real operator logs
- Add visualization (timeline charts)
- Use LSTM autoencoder to detect anomaly
- Use BART-Large/T5 for insight generation
-  Discretized numeric sensor values (e.g., temperature, pressure) into qualitative levels like "high" or "normal" to improve semantic similarity with unstructured operator logs.
- Improve log diversity
- Integrate real-time streaming data
