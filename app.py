#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from tsaug import AddNoise
from sklearn.ensemble import IsolationForest
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

# Load Hugging Face summarization pipeline (free model)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


# In[61]:


embedder = SentenceTransformer("all-MiniLM-L6-v2")


# In[38]:


def generate_data():
    time = pd.date_range(datetime.now() - timedelta(days=90), periods=2160, freq='H')
    temp = AddNoise(scale=0.1).augment(np.random.normal(75, 2, size=len(time)))
    pressure = AddNoise(scale=0.1).augment(np.random.normal(30, 1, size=len(time)))

    df = pd.DataFrame({'timestamp': time, 'temperature': temp, 'pressure': pressure})

    # Inject missing values randomly
    for col in ['temperature', 'pressure']:
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, col] = np.nan

    return df


# In[41]:


# 2. Detect anomalies using Isolation Forest
def detect_anomalies(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    features = df[['temperature', 'pressure']].copy()
    features = features.fillna(method='ffill')  # fill missing values before fitting
    df['anomaly'] = model.fit_predict(features)
    return df[df['anomaly'] == -1]


# In[35]:


def generate_logs(df):
    logs = df.iloc[::50][['timestamp']].copy()
    logs['log'] = 'Operator noticed unusual reading'
    return logs


# In[55]:


from sentence_transformers import SentenceTransformer, util


# In[67]:


# 4. Correlate anomalies with logs using cosine similarity of embeddings


def map_to_level(value, feature):
    if feature == "temperature":
        return "high temperature" if value > 90 else "normal temperature"
    elif feature == "pressure":
        return "high pressure" if value > 50 else "normal pressure"
    return f"{feature} value"

def correlate(anomalies, logs):
    correlated = []
    log_embeddings = embedder.encode(logs['log'].tolist(), convert_to_tensor=True)

    for _, a_row in anomalies.iterrows():
        temp_level = map_to_level(a_row['temperature'], 'temperature')
        pressure_level = map_to_level(a_row['pressure'], 'pressure')
        anomaly_text = f"Detected {temp_level} and {pressure_level} at {a_row['timestamp']}"
        anomaly_embedding = embedder.encode(anomaly_text, convert_to_tensor=True)

        # Filter logs to those within a 1-hour window of the anomaly
        close_logs = logs[np.abs((logs['timestamp'] - a_row['timestamp']).dt.total_seconds()) <= 3600]

        if close_logs.empty:
            best_log = {'timestamp': None, 'log': 'No nearby log found'}
            best_score = 0.0
        else:
            close_log_embeddings = embedder.encode(close_logs['log'].tolist(), convert_to_tensor=True)
            cosine_scores = util.cos_sim(anomaly_embedding, close_log_embeddings)[0].cpu().numpy()
            best_idx = np.argmax(cosine_scores)
            best_score = cosine_scores[best_idx]
            best_log = close_logs.iloc[best_idx]

        correlated.append({
            'anomaly_time': a_row['timestamp'],
            'temperature': a_row['temperature'],
            'pressure': a_row['pressure'],
            'anomaly_text': anomaly_text,
            'log_time': best_log['timestamp'],
            'log': best_log['log'],
            'similarity_score': best_score
        })

    return pd.DataFrame(correlated)


# In[70]:


# 5. Generate insight using Hugging Face summarizer
def generate_insight(row):
    prompt = f"Sensor anomaly at {row['anomaly_time']} with temperature {row['temperature']:.2f} and pressure {row['pressure']:.2f}. Operator noted: {row['log']}."
    try:
        summary = summarizer(prompt, max_length=50, min_length=20, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {e}"


# In[74]:


def main():
    st.title("Chevron Anomaly Demo")
    df = generate_data()
    st.subheader("Full Sensor Data")
    st.write(df)

    anomalies = detect_anomalies(df)
    st.subheader("Anomalies")
    st.write(anomalies)

    logs = generate_logs(df)
    st.subheader("Logs")
    st.write(logs)

    correlated = correlate(anomalies, logs)
    st.subheader("Correlated Anomalies and Logs with Similarity Score")
    st.write(correlated)

    st.subheader("GenAI Insights")
    for i in range(min(3, len(correlated))):
        row = correlated.iloc[i]
        insight = generate_insight(row)
        st.markdown(f"**Insight {i+1}:** {insight}")

if __name__ == '__main__':
    main()


# In[12]:



