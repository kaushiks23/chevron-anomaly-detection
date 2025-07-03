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
    temp = AddNoise(scale=10).augment(np.random.normal(75, 2, size=len(time)))
    pressure = AddNoise(scale=5).augment(np.random.normal(30, 1, size=len(time)))

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
def correlatev2(anomalies, logs):
    correlatedv2 = []
    log_embeddings = embedder.encode(logs['log'].tolist(), convert_to_tensor=True)

    for _, a_row in anomalies.iterrows():
        anomaly_text = f"Spike detected at {a_row['timestamp']} with temperature {a_row['temperature']:.2f} and pressure {a_row['pressure']:.2f}"
        anomaly_embedding = embedder.encode(anomaly_text, convert_to_tensor=True)

        cosine_scores = util.cos_sim(anomaly_embedding, log_embeddings)[0].cpu().numpy()
        best_idx = np.argmax(cosine_scores)
        best_score = cosine_scores[best_idx]

        correlatedv2.append({
            'anomaly_time': a_row['timestamp'],
            'temperature': a_row['temperature'],
            'anomaly_text': anomaly_text,
            'pressure': a_row['pressure'],
            'log_time': logs.iloc[best_idx]['timestamp'],
            'log': logs.iloc[best_idx]['log'],
            'similarity_score': best_score

        })

    return pd.DataFrame(correlatedv2)


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

    correlatedv2 = correlatev2(anomalies, logs)
    st.subheader("Correlated Anomalies and Logs with Similarity Score")
    st.write(correlatedv2)

    st.subheader("GenAI Insights")
    for i in range(min(3, len(correlatedv2))):
        row = correlatedv2.iloc[i]
        insight = generate_insight(row)
        st.markdown(f"**Insight {i+1}:** {insight}")

if __name__ == '__main__':
    main()


# In[12]:



