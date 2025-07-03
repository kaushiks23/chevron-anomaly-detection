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

- Used semantic similarity (text meaning) between anomaly descriptions and operator logs, instead of relying solely on timestamp matching, to better capture meaningful correlations even when logs and anomalies are slightly misaligned in time
- Chose Isolation Forest over LSTM Autoencoder due to its simplicity, speed, and minimal data requirements
- Selected lightweight models (free-tier friendly)
- Streamlit app for fast prototyping

## Failure Points

- No ground-truth to evaluate anomaly detection
- Isolation forest contamination is set at 5%. Can lead to false positives/negatives
- If anomaly is repetitive, isolation forest may miss it
- If there is a gradual pressure increase over time, isolation forest may miss it
- Using only limited features (temperature,pressure)

## Future Work

- Look at a larger timeframe
- Add visualization (timeline charts)
- Optimize Isolation Forest by hyperparameter tuning and standardizing/normalising features
- Experiment with LSTM autoencoder to detect anomaly
- Experiment with BART-Large/T5 for insight generation
- Improve log diversity
