import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import random
import re

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
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.png", width=100)
with col2:
    st.markdown("<div class='typing-header'>Airline Sentiment Analyzer by Vikrant</div>", unsafe_allow_html=True)

# 📘 Sidebar Branding
with st.sidebar:
    st.header("📘 About")
    st.markdown("""
    This dashboard helps **airline customer experience teams** monitor passenger sentiment using NLP.  
    It analyzes feedback across carriers and visualizes trends to support **CX decisions**, **route planning**, and **service recovery**.
    """)
    st.info("📌 Tip: Upload a CSV with a column like 'text', 'review', or 'comments' containing customer feedback.")

# 📂 File Upload
st.markdown("### 📄 Upload Your Own CSV")
uploaded_file = st.file_uploader("Upload airline-reviews.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom file uploaded successfully.")
else:
    st.error("❌ Default file not found. Please upload a CSV file.")
    st.stop()

st.write("📁 Active file:", uploaded_file.name if uploaded_file else "N/A")

# ✈️ Simulate airline column if missing
if "airline" not in df.columns:
    df["airline"] = [random.choice(
        ["Indigo", "Air India", "SpiceJet", "Vistara", "Akasa", "Air Asia"]
    ) for _ in range(len(df))]

# 🧠 Sentiment Analysis
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    huggingface_available = True
except Exception as e:
    st.warning("⚠️ Hugging Face model failed. Using VADER fallback.")
    huggingface_available = False
    vader_analyzer = SentimentIntensityAnalyzer()

# Select text column
text_candidates = [col for col in df.columns if df[col].dtype == "object" and df[col].str.len().mean() > 30]
default_text_col = "text" if "text" in df.columns else (text_candidates[0] if text_candidates else None)

if default_text_col:
    st.markdown("### 📝 Select Text Column for Sentiment Analysis")
    selected_text_col = st.selectbox("Choose column containing customer feedback:", df.columns, index=df.columns.get_loc(default_text_col))
else:
    st.error("❌ No suitable text column found. Please upload a CSV with a column like 'text', 'review', or 'comments'.")
    st.stop()

# Apply sentiment
def get_sentiment(text):
    if huggingface_available:
        try:
            return sentiment_pipeline(str(text))[0]["label"].upper()
        except:
            # fallback to VADER if HF fails
            score = vader_analyzer.polarity_scores(str(text))['compound']
    else:
        score = vader_analyzer.polarity_scores(str(text))['compound']
    return "POSITIVE" if score >= 0.05 else "NEGATIVE" if score <= -0.05 else "NEUTRAL"

df["sentiment"] = df[selected_text_col].apply(get_sentiment)

# ✈️ Normalize airline names
df["airline"] = df["airline"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()
airline_map = {"Air Asia": "AirAsia", "Air Asia India": "AirAsia", "Akasa Air": "Akasa", "Akasa Airlines": "Akasa"}
df["airline"] = df["airline"].replace(airline_map)

# ✈️ Airline Filter
selected_airline = st.selectbox("✈️ Filter by Airline", sorted(df["airline"].unique()))
df = df[df["airline"] == selected_airline]

# 📈 Sentiment Trend Over Time (Stacked Area Chart)
st.markdown("### 📈 Sentiment Trend Over Time")
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df = df[df["date"].notna()]
    if not df.empty:
        trend_df = df.groupby(["date", "sentiment"]).size().reset_index(name="count")
        fig_trend = px.area(trend_df, x="date", y="count", color="sentiment",
                            color_discrete_map={"POSITIVE": "blue", "NEGATIVE": "crimson", "NEUTRAL": "grey"},
                            title=f"Sentiment Trend for {selected_airline}")
        st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No date column found. Trendline skipped.")

# 📊 Diverging Sentiment Heatmap
st.markdown("### 📊 Diverging Sentiment Heatmap")
if "date" in df.columns and not df.empty:
    heat_df = df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)
    heat_df = heat_df.reindex(columns=["POSITIVE", "NEGATIVE", "NEUTRAL"], fill_value=0)
    fig_heat = px.imshow(heat_df.T, labels=dict(x="Date", y="Sentiment", color="Count"),
                         x=heat_df.index, y=heat_df.columns,
                         color_continuous_scale="RdBu", aspect="auto",
                         title=f"Diverging Sentiment Heatmap for {selected_airline}")
    st.plotly_chart(fig_heat, use_container_width=True)

# 📊 Sentiment Distribution
st.markdown("### 📊 Sentiment Distribution")
sentiment_counts = df["sentiment"].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90,
        colors=['#66b3ff', '#ff6666', '#ffcc99'])
ax2.axis('equal')
st.pyplot(fig2, use_container_width=True)

# 🧠 Word Cloud for Negative Sentiment
st.markdown("### 🧠 Frequent Negative Keywords")
custom_stopwords = set(STOPWORDS).union(["positive","negative","neutral","POSITIVE","NEGATIVE","NEUTRAL",
                                         "experience","service","flight","airline","good","bad","okay","delay","delayed","late","on","off","get","got"])
neg_text_series = df[df["sentiment"] == "NEGATIVE"][selected_text_col].dropna().astype(str)
tokens = []
for text in neg_text_series:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    tokens.extend([w for w in words if w not in custom_stopwords])
neg_text = " ".join(tokens)
if neg_text.strip():
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=custom_stopwords, collocations=False, max_words=100).generate(neg_text)
    fig_wc, ax_wc = plt.subplots(figsize=(10,5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc, use_container_width=True)
else:
    st.info("No negative sentiment found for this airline.")

# ⚠️ CX Alert
st.markdown("### ⚠️ CX Alert")
neg_count = sentiment_counts.get("NEGATIVE", 0)
if neg_count > 10:
    st.error(f"Spike in negative sentiment detected for {selected_airline}. Investigate recent feedback.")
else:
    st.success("No major negative sentiment spike detected.")

# 📌 Footer
st.markdown("---")
st.markdown("**✈️ From Runways to Regression Models — Aviation Expertise Meets Data Intelligence.**")
st.markdown("""
### 🔗 Connect with Me  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-vthenge-blue?logo=linkedin)](https://www.linkedin.com/in/vthenge)  
[![Email](https://img.shields.io/badge/Email-vikrantthenge@outlook.com-red?logo=gmail)](mailto:vikrantthenge@outlook.com)  
[![GitHub](https://img.shields.io/badge/GitHub-vikrantthenge-black?logo=github)](https://github.com/vikrantthenge)
""", unsafe_allow_html=True)

st.markdown("[🚀 Live App](https://sentiment-analyzer-vikrant.streamlit.app)")
