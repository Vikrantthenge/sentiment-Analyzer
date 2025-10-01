import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
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

# 📂 File Upload or Default
st.markdown("### 📄 Upload Your Own CSV or Use Default Demo File")
uploaded_file = st.file_uploader("Upload airline-reviews.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom file uploaded successfully.")
else:
    df = pd.read_csv("airline_feedback.csv")
    st.info("ℹ️ Using default demo file: airline_feedback.csv")

st.write("📁 Active file:", uploaded_file.name if uploaded_file else "airline_feedback.csv")

# 🧠 Sentiment Analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

text_candidates = [col for col in df.columns if df[col].dtype == "object" and df[col].str.len().mean() > 30]
default_text_col = "text" if "text" in df.columns else (text_candidates[0] if text_candidates else None)

if default_text_col:
    st.markdown("### 📝 Select Text Column for Sentiment Analysis")
    selected_text_col = st.selectbox("Choose column containing customer feedback:", df.columns, index=df.columns.get_loc(default_text_col))

    try:
        df["sentiment"] = df[selected_text_col].apply(lambda x: sentiment_pipeline(str(x))[0]["label"].upper())
    except Exception as e:
        st.error("❌ Error applying sentiment analysis. Please check if the selected column contains valid text.")
        st.exception(e)
        st.stop()
else:
    st.error("❌ No suitable text column found. Please upload a CSV with a column like 'text', 'review', or 'comments'.")
    st.stop()

# ✈️ Ensure airline column includes full carrier list
airline_list = ["Indigo", "Air India", "SpiceJet", "Vistara", "Akasa", "AirAsia"]

if "airline" not in df.columns:
    df["airline"] = [random.choice(airline_list) for _ in range(len(df))]
else:
    existing_airlines = df["airline"].dropna().unique().tolist()
    missing_airlines = [air for air in airline_list if air not in existing_airlines]
    if missing_airlines:
        filler_rows = pd.DataFrame({
            "airline": missing_airlines,
            selected_text_col: [""] * len(missing_airlines),
            "sentiment": ["POSITIVE"] * len(missing_airlines)
        })
        df = pd.concat([df, filler_rows], ignore_index=True)

# 📈 Sentiment Trend Over Time
st.markdown("### 📈 Sentiment Trend Over Time")

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df = df[df["date"].notna()]
    df_trend = df.copy()

    grouped = df_trend.groupby(["date", "sentiment"]).size().reset_index(name="count")
    pivot_df = grouped.pivot(index="date", columns="sentiment", values="count").fillna(0)
    pivot_df = pivot_df.reset_index().melt(id_vars="date", var_name="sentiment", value_name="count")

    if pivot_df.empty:
        st.info("Not enough data to show sentiment trend.")
    else:
        fig_trend = px.line(
            pivot_df,
            x="date",
            y="count",
            color="sentiment",
            title="Sentiment Over Time",
            color_discrete_map={
                "POSITIVE": "blue",
                "NEGATIVE": "crimson"
            }
        )
        fig_trend.update_traces(mode="lines+markers")
        fig_trend.update_layout(
            legend_title_text="Sentiment",
            yaxis_title="Count",
            xaxis_title="Date"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No date column found. Trendline skipped.")

# 📊 Diverging Bar Chart
st.markdown("### 📊 Diverging Sentiment Bar Chart")

div_df = df_trend.groupby(["date", "sentiment"]).size().unstack(fill_value=0).reset_index()
div_df["POSITIVE"] = div_df.get("POSITIVE", 0)
div_df["NEGATIVE"] = -div_df.get("NEGATIVE", 0)

div_melted = div_df.melt(id_vars="date", value_vars=["POSITIVE", "NEGATIVE"], var_name="sentiment", value_name="count")

fig_diverge = px.bar(
    div_melted,
    x="date",
    y="count",
    color="sentiment",
    title="Diverging Sentiment by Date",
    color_discrete_map={"POSITIVE": "blue", "NEGATIVE": "red"},
    barmode="relative"
)
fig_diverge.update_layout(yaxis_title="Sentiment Count", xaxis_title="Date")
st.plotly_chart(fig_diverge)

# ✈️ Airline Filter
# ✈️ Ensure airline column includes full carrier list
airline_list = ["Indigo", "Air India", "SpiceJet", "Vistara", "Akasa", "AirAsia"]

if "airline" not in df.columns:
    df["airline"] = [random.choice(airline_list) for _ in range(len(df))]
else:
    existing_airlines = df["airline"].dropna().unique().tolist()
    missing_airlines = [air for air in airline_list if air not in existing_airlines]
    if missing_airlines:
        filler_rows = pd.DataFrame({
            "airline": missing_airlines,
            selected_text_col: [""] * len(missing_airlines),
            "sentiment": ["POSITIVE"] * len(missing_airlines)
        })
        df = pd.concat([df, filler_rows], ignore_index=True)
        
selected_airline = st.selectbox("✈️ Filter by Airline", df["airline"].unique())
df = df[df["airline"] == selected_airline]

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

custom_stopwords = set(STOPWORDS)
custom_stopwords.update([
    "positive", "negative", "neutral",
    "POSITIVE", "NEGATIVE", "NEUTRAL",
    "experience", "service", "flight", "airline",
    "good", "bad", "okay", "delay", "delayed", "late", "on", "off", "get", "got"
])

neg_text_series = df[df["sentiment"] == "NEGATIVE"][selected_text_col].dropna().astype(str)

tokens = []
for text in neg_text_series:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered = [word for word in words if word not in custom_stopwords]
    tokens.extend(filtered)

neg_text = " ".join(tokens)

if neg_text.strip():
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=custom_stopwords,
        collocations=False,
        max_words=100
    ).generate(neg_text)

    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc, use_container_width=True)
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

