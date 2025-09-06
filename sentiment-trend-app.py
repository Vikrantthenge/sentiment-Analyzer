# ✈️ Airline Sentiment Analyzer by Vikrant Thenge

import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random

# 🌐 Page Config
st.set_page_config(page_title="✈️ Airline Sentiment Analyzer", layout="centered")

# 🖼️ Logo + Animated Header
st.markdown("""
<style>
@keyframes typing { from { width: 0 } to { width: 100% } }
@keyframes blink { 50% { border-color: transparent } }
.typing-header {
  font-size: 32px; font-weight: bold; white-space: nowrap;
  overflow: hidden; border-right: 3px solid #0078D4;
  width: 0; animation: typing 3s steps(30, end) forwards, blink 0.75s step-end infinite;
  color: #0078D4; margin-bottom: 20px;
}
.glow-line {
  height: 2px; background: linear-gradient(90deg, #0078D4, #00B4FF);
  animation: glow 2s infinite alternate; margin: 20px 0;
}
.contact-strip {
  display: flex; align-items: center; gap: 20px; font-size: 16px; margin-top: 10px;
}
.contact-strip a { text-decoration: none; color: #333; }
.contact-strip img { vertical-align: middle; margin-right: 5px; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.png", width=100)
with col2:
    st.markdown("<div class='typing-header'>Airline Sentiment Analyzer by Vikrant</div>", unsafe_allow_html=True)

# ✈️ Aviation Context Introduction
st.markdown("""
### 📘 About Section
This dashboard helps **airline customer experience teams** monitor passenger sentiment using NLP.  
It analyzes feedback across carriers and visualizes trends to support **CX decisions**, **route planning**, and **service recovery**.
""")

# 📂 File Upload or Default
st.markdown("### 📄 Upload Your Own CSV or Use Default Demo File")
uploaded_file = st.file_uploader("Upload airline_reviews.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom file uploaded successfully.")
else:
    df = pd.read_csv("airline_feedback.csv")
    st.info("ℹ️ Using default demo file: airline_feedback.csv")

# 📁 Show active file name
st.write("📁 Active file:", uploaded_file.name if uploaded_file else "airline_feedback.csv")

# ✈️ Simulate airline column if missing
if "airline" not in df.columns:
    df["airline"] = [random.choice(["Indigo", "Air India", "SpiceJet", "Vistara"]) for _ in range(len(df))]

# 🧠 Sentiment Analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
df["sentiment"] = df["text"].apply(lambda x: sentiment_pipeline(x)[0]["label"].upper())

# ✈️ Airline Filter
selected_airline = st.selectbox("✈️ Filter by Airline", df["airline"].unique())
df = df[df["airline"] == selected_airline]

# 📊 Sentiment Distribution
st.markdown("### 📊 Sentiment Distribution")
sentiment_counts = df["sentiment"].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90,
        colors=['#66b3ff', '#99ff99', '#ffcc99'])
ax2.axis('equal')
st.pyplot(fig2, use_container_width=True)

# 📈 Sentiment Trend Over Time
st.markdown("### 📈 Sentiment Trend Over Time")
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    trend_df = df.groupby(["date", "sentiment"]).size().reset_index(name="count")
    fig_trend = px.line(trend_df, x="date", y="count", color="sentiment", title="Sentiment Over Time")
    st.plotly_chart(fig_trend)
else:
    st.info("No date column found. Trendline skipped.")

# 🧠 Word Cloud for Negative Sentiment
st.markdown("### 🧠 Frequent Negative Keywords")
neg_text = " ".join(df[df["sentiment"] == "NEGATIVE"]["text"])
if neg_text:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(neg_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
else:
    st.info("No negative sentiment found for this airline.")

# ⚠️ CX Alert Section
st.markdown("### ⚠️ CX Alert")
neg_count = sentiment_counts.get("NEGATIVE", 0)
if neg_count > 10:
    st.error(f"Spike in negative sentiment detected for {selected_airline}. Investigate recent feedback.")
else:
    st.success("No major negative sentiment spike detected.")

# 📌 Footer Branding with Badges
st.markdown("---")
st.markdown("**✈️ From Runways to Regression Models — Aviation Expertise Meets Data Intelligence.**")
st.markdown("""
### 🔗 Connect with Me  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-vthenge-blue?logo=linkedin)](https://www.linkedin.com/in/vthenge)  
[![Email](https://img.shields.io/badge/Email-vikrantthenge@outlook.com-red?logo=gmail)](mailto:vikrantthenge@outlook.com)  
[![GitHub](https://img.shields.io/badge/GitHub-vikrantthenge-black?logo=github)](https://github.com/vikrantthenge)
""", unsafe_allow_html=True)

st.markdown("[🚀 Live App](https://sentiment-analyzer-vikrant.streamlit.app)")


