# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import random
import re
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# 🌐 Page Config
# -------------------------
st.set_page_config(page_title="✈️ Airline Sentiment Analyzer", layout="centered")

# -------------------------
# 🖼️ Header + Logo + Animated Title
# -------------------------
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 5])
with col1:
    # Keep logo.png in the same folder as this script or comment this line if not available
    try:
        st.image("logo.png", width=100)
    except Exception:
        # fallback: show nothing if logo not found
        st.write("")
with col2:
    st.markdown("<div class='typing-header'>Airline Sentiment Analyzer by Vikrant</div>", unsafe_allow_html=True)

# -------------------------
# 🔀 Mode Selection
# -------------------------
mode = st.radio("Choose Mode", ["Basic Sentiment", "NLP Pipeline Demo"])

# -------------------------
# NLP Pipeline Demo (spaCy)
# -------------------------
if mode == "NLP Pipeline Demo":
    st.subheader("🧬 NLP Pipeline Demo (spaCy)")
    user_input = st.text_area("Enter text for NLP processing")

    if user_input:
        # lazy import + download if missing
        try:
            import spacy
        except Exception as e:
            st.error("spaCy is not installed. Install it (`pip install spacy`) to use the NLP demo.")
            st.stop()

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # try programmatic download (may not work in some restricted environments)
            from spacy.cli import download

            with st.spinner("Downloading spaCy model (en_core_web_sm)..."):
                try:
                    download("en_core_web_sm")
                    nlp = spacy.load("en_core_web_sm")
                except Exception as e:
                    st.error("Failed to download/load spaCy model. Please install `python -m spacy download en_core_web_sm` locally.")
                    st.exception(e)
                    st.stop()

        doc = nlp(user_input)

        st.markdown("**🔤 Tokens:**")
        st.write([token.text for token in doc])

        st.markdown("**🧾 Lemmas:**")
        st.write([token.lemma_ for token in doc])

        st.markdown("**🏷️ Named Entities:**")
        st.write([(ent.text, ent.label_) for ent in doc.ents])

        st.markdown("**📊 POS Tags:**")
        st.write([(token.text, token.pos_) for token in doc])

    st.stop()

# -------------------------
# 📘 Sidebar Branding / Help
# -------------------------
with st.sidebar:
    st.header("📘 About")
    st.markdown(
        """
    This dashboard helps **airline customer experience teams** monitor passenger sentiment using NLP.  
    It analyzes feedback across carriers and visualizes trends to support **CX decisions**, **route planning**, and **service recovery**.
    """
    )
    st.info("📌 Tip: Upload a CSV with a column like 'text', 'review', or 'comments' containing customer feedback.")

# -------------------------
# 📂 File Upload or Default
# -------------------------
st.markdown("### 📄 Upload Your Own CSV or Use Default Demo File")
DEFAULT_CSV_URL = "https://raw.githubusercontent.com/Vikrantthenge/sentiment-Analyzer/main/airline-reviews.csv"

uploaded_file = st.file_uploader("Upload airline-reviews.csv", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Custom file uploaded successfully.")
    except Exception as e:
        st.error("Could not read the uploaded CSV. Please ensure it's a valid CSV file.")
        st.exception(e)
        st.stop()
else:
    # load default (may fail if offline)
    try:
        df = pd.read_csv(DEFAULT_CSV_URL)
        st.info("ℹ️ Using default demo file from GitHub")
    except Exception:
        st.error("❌ Default file not found. Please upload a CSV file.")
        st.stop()

st.write("📁 Active file:", uploaded_file.name if uploaded_file else "airline-reviews.csv")

# -------------------------
# ✈️ Ensure airline column exists
# -------------------------
if "airline" not in df.columns:
    df["airline"] = [
        random.choice(["Indigo", "Air India", "SpiceJet", "Vistara", "Akasa", "Air Asia"])
        for _ in range(len(df))
    ]

# -------------------------
# 🧠 Sentiment Analysis setup (Hugging Face -> fallback VADER)
# -------------------------
huggingface_available = False
analyzer = None
sentiment_pipeline = None

try:
    with st.spinner("Loading Hugging Face sentiment model..."):
        sentiment_pipeline = pipeline(
            "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        huggingface_available = True
except Exception:
    st.warning("⚠️ Hugging Face model failed to load. Switching to VADER fallback...")
    analyzer = SentimentIntensityAnalyzer()
    huggingface_available = False

# -------------------------
# 🔎 Auto-detect candidate text columns
# -------------------------
# Candidate object dtype columns with an average length heuristic
text_candidates = [
    col
    for col in df.columns
    if df[col].dtype == "object"
    and df[col].dropna().astype(str).map(len).mean() >= 20
]

# Prefer explicit common names
default_text_col = None
for name in ("text", "review", "comments", "feedback"):
    if name in df.columns:
        default_text_col = name
        break
if default_text_col is None and text_candidates:
    default_text_col = text_candidates[0]

if default_text_col is None:
    st.error("❌ No suitable text column found. Please upload a CSV with a column like 'text', 'review', or 'comments'.")
    st.stop()

st.markdown("### 📝 Select Text Column for Sentiment Analysis")
selected_text_col = st.selectbox("Choose column containing customer feedback:", df.columns.tolist(), index=df.columns.get_loc(default_text_col))

# Ensure the selected column is cast to string for processing (avoid NaNs)
df[selected_text_col] = df[selected_text_col].astype(str).fillna("")

# -------------------------
# Apply Sentiment Analysis (vectorized where possible)
# -------------------------
try:
    if huggingface_available and sentiment_pipeline is not None:
        # apply in batches to be safer on memory and to show progress
        results = []
        batch_size = 64
        texts = df[selected_text_col].fillna("").astype(str).tolist()
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            preds = sentiment_pipeline(batch)
            # each pred is like {'label': 'POSITIVE', 'score': 0.999}
            results.extend([p.get("label", "NEUTRAL").upper() for p in preds])
        df["sentiment"] = results
    else:
        # VADER fallback
        analyzer = analyzer or SentimentIntensityAnalyzer()

        def vader_sentiment(text: str) -> str:
            score = analyzer.polarity_scores(str(text))["compound"]
            if score >= 0.05:
                return "POSITIVE"
            elif score <= -0.05:
                return "NEGATIVE"
            else:
                return "NEUTRAL"

        df["sentiment"] = df[selected_text_col].apply(vader_sentiment)
except Exception as e:
    st.error("❌ Error applying sentiment analysis. Please check if the selected column contains valid text.")
    st.exception(e)
    st.stop()

# -------------------------
# Normalize airline names
# -------------------------
df["airline"] = df["airline"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()

# manual mapping
airline_map = {
    "Air Asia": "AirAsia",
    "Air Asia India": "AirAsia",
    "Akasa Air": "Akasa",
    "Akasa Airlines": "Akasa",
}
df["airline"] = df["airline"].replace(airline_map)

# -------------------------
# Airline filter
# -------------------------
airlines_sorted = sorted(df["airline"].unique())
selected_airline = st.selectbox("✈️ Filter by Airline", airlines_sorted)
df = df[df["airline"] == selected_airline].copy()

if df.empty:
    st.warning("No rows available for the selected airline after filtering.")
    st.stop()

# -------------------------
# 📊 Sentiment Volume by Airline (grouped bar)
# -------------------------
st.markdown("### 📊 Sentiment Volume (Selected Airline)")
sentiment_airline = df.groupby(["airline", "sentiment"]).size().reset_index(name="count")

# ensure the sentiments are in consistent order
sentiment_airline["sentiment"] = sentiment_airline["sentiment"].astype(str).str.upper()
sentiment_order = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

fig_grouped = px.bar(
    sentiment_airline,
    x="airline",
    y="count",
    color="sentiment",
    category_orders={"sentiment": sentiment_order},
    barmode="group",
    title="Sentiment Volume",
    color_discrete_map={"POSITIVE": "blue", "NEGATIVE": "red", "NEUTRAL": "gray"},
)
st.plotly_chart(fig_grouped, use_container_width=True)

# -------------------------
# 📈 Sentiment Trend Over Time (if date exists)
# -------------------------
st.markdown("### 📈 Sentiment Trend Over Time")
if "date" in df.columns:
    # try several common date formats
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    if df["date"].isna().all():
        # try dayfirst
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    df = df[df["date"].notna()].copy()
    if df.empty:
        st.info("No valid dates found in the 'date' column to plot trends.")
    else:
        df_trend = df.copy()
        grouped = df_trend.groupby(["date", "sentiment"]).size().reset_index(name="count")
        pivot_df = grouped.pivot(index="date", columns="sentiment", values="count").fillna(0)
        pivot_melt = pivot_df.reset_index().melt(id_vars="date", var_name="sentiment", value_name="count")
        if pivot_melt.empty:
            st.info("Not enough data to show sentiment trend.")
        else:
            fig_trend = px.area(
                pivot_melt,
                x="date",
                y="count",
                color="sentiment",
                title="Sentiment Over Time",
                color_discrete_map={"POSITIVE": "blue", "NEGATIVE": "crimson", "NEUTRAL": "gray"},
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # rolling average (7-day) - only for dense date series
            st.markdown("### 📈 Smoothed Sentiment Trend (7-Day Rolling Avg)")
            rolling_df = (
                df.groupby(["date", "sentiment"]).size().unstack(fill_value=0).sort_index()
            )
            rolling_avg = rolling_df.rolling(window=7, min_periods=1).mean().reset_index().melt(
                id_vars="date", var_name="sentiment", value_name="count"
            )
            fig_smooth = px.line(
                rolling_avg,
                x="date",
                y="count",
                color="sentiment",
                title="7-Day Rolling Average of Sentiment",
                color_discrete_map={"POSITIVE": "blue", "NEGATIVE": "red", "NEUTRAL": "gray"},
            )
            st.plotly_chart(fig_smooth, use_container_width=True)

            # Sentiment heatmap by weekday
            st.markdown("### 📍 Sentiment Heatmap by Weekday")
            df["weekday"] = df["date"].dt.day_name()
            heatmap_df = df.groupby(["weekday", "sentiment"]).size().unstack(fill_value=0)
            # Ensure consistent weekday order
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heatmap_df = heatmap_df.reindex(weekday_order).fillna(0)
            fig_heatmap = px.imshow(
                heatmap_df,
                labels=dict(x="Sentiment", y="Weekday", color="Count"),
                title="Sentiment Volume by Weekday",
                color_continuous_scale="RdBu",
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Diverging sentiment bar by date
            st.markdown("### 📊 Diverging Sentiment by Date")
            div_df = df.groupby(["date", "sentiment"]).size().unstack(fill_value=0).reset_index()
            div_df["POSITIVE"] = div_df.get("POSITIVE", 0)
            div_df["NEGATIVE"] = -div_df.get("NEGATIVE", 0)
            # melt only POSITIVE and NEGATIVE (neutral omitted for diverging)
            div_melted = div_df.melt(id_vars="date", value_vars=["POSITIVE", "NEGATIVE"], var_name="sentiment", value_name="count")
            fig_diverge = px.bar(
                div_melted,
                y="date",
                x="count",
                color="sentiment",
                title="Diverging Sentiment by Date",
                color_discrete_map={"POSITIVE": "blue", "NEGATIVE": "red"},
                barmode="relative",
            )
            fig_diverge.update_layout(yaxis_title="Date", xaxis_title="Sentiment Count")
            st.plotly_chart(fig_diverge, use_container_width=True)

            # Radar chart (profile) — only meaningful if multiple airlines present; here we show for the selected airline categories
            st.markdown("### 🧭 Airline Sentiment Profile (Radar Chart)")
            radar_df = df.groupby(["airline", "sentiment"]).size().unstack(fill_value=0).reset_index()
            try:
                radar_long = radar_df.melt(id_vars="airline", var_name="sentiment", value_name="count")
                fig_radar = px.line_polar(
                    radar_long,
                    r="count",
                    theta="sentiment",
                    color="airline",
                    line_close=True,
                    title="Sentiment Profile by Airline",
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            except Exception:
                # skip radar if data shape is incompatible
                pass

# -------------------------
# 📊 Sentiment Distribution (pie)
# -------------------------
st.markdown("### 📊 Sentiment Distribution")
sentiment_counts = df["sentiment"].value_counts().reindex(["POSITIVE", "NEUTRAL", "NEGATIVE"]).fillna(0)
fig2, ax2 = plt.subplots()
colors = ["#66b3ff", "#ffcc99", "#ff6666"]  # order: positive, neutral, negative
ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90, colors=colors)
ax2.axis("equal")
st.pyplot(fig2, use_container_width=True)

# -------------------------
# 🧠 Word Cloud for Negative Sentiment
# -------------------------
st.markdown("### 🧠 Frequent Negative Keywords")
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(
    [
        "positive",
        "negative",
        "neutral",
        "POSITIVE",
        "NEGATIVE",
        "NEUTRAL",
        "experience",
        "service",
        "flight",
        "airline",
        "good",
        "bad",
        "okay",
        "delay",
        "delayed",
        "late",
        "on",
        "off",
        "get",
        "got",
    ]
)

neg_text_series = df[df["sentiment"] == "NEGATIVE"][selected_text_col].dropna().astype(str)
tokens = []
for text in neg_text_series:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    filtered = [word for word in words if word not in custom_stopwords]
    tokens.extend(filtered)

neg_text = " ".join(tokens)
if neg_text.strip():
    wordcloud = WordCloud(width=800, height=400, background_color="white", stopwords=custom_stopwords, collocations=False, max_words=100).generate(neg_text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc, use_container_width=True)
else:
    st.info("No negative sentiment found for this airline.")

# -------------------------
# ⚠️ CX Alert Section
# -------------------------
st.markdown("### ⚠️ CX Alert")
neg_count = int(sentiment_counts.get("NEGATIVE", 0))
if neg_count > 10:
    st.error(f"🚨 Spike in negative sentiment detected for {selected_airline}. Investigate recent feedback.")
else:
    st.success("✅ No major negative sentiment spike detected.")

# -------------------------
# 📌 Footer Branding
# -------------------------
st.markdown("---")
st.markdown("**✈️ From Runways to Regression Models — Aviation Expertise Meets Data Intelligence.**")
st.markdown(
    """
### 🔗 Connect with Me  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-vthenge-blue?logo=linkedin)](https://www.linkedin.com/in/vthenge)  
[![Email](https://img.shields.io/badge/Email-vikrantthenge@outlook.com-red?logo=gmail)](mailto:vikrantthenge@outlook.com)  
[![GitHub](https://img.shields.io/badge/GitHub-vikrantthenge-black?logo=github)](https://github.com/vikrantthenge)
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style='text-align: center; font-size:16px; font-weight:normal; color:#343a40; line-height:1.2;'>
🔍 Powered by NLP & CX Intelligence — Built for Airline Feedback Precision.<br>
🔐 This dashboard analyzes anonymized feedback only. No personal data is stored.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style='text-align: center; font-size: 16px; font-weight: bold; color: #000000;'>
🛠️ Version: v1.0 | 📅 Last Updated: October 2025
</div>
""",
    unsafe_allow_html=True,
)
