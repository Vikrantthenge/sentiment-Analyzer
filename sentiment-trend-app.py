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

    # 🔀 Mode Selection: Basic vs NLP Pipeline
import streamlit as st
import spacy
from spacy.cli import download

def main():
    import streamlit as st
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# 🔀 Mode Selection
mode = st.radio("Choose Mode", ["Basic Sentiment", "NLP Pipeline Demo"])

if mode == "NLP Pipeline Demo":
    st.subheader("🧬 NLP Pipeline Output")
    user_input = st.text_area("Enter text for NLP processing")

    try:
        nlp = spacy.load("./en_core_web_sm/en_core_web_sm-3.8.0")
    except OSError:
        st.error("⚠️ spaCy model not found. Please ensure it's bundled correctly.")
        st.stop()

    if user_input:
     doc = nlp(user_input)

    # 🧠 Emoji Mapping for Entity Types
    ENTITY_EMOJI_MAP = {
        "PERSON": "🧑",
        "ORG": "🏢",
        "GPE": "🌍",
        "LOC": "📍",
        "DATE": "📅",
        "TIME": "⏰",
        "MONEY": "💰",
        "QUANTITY": "🔢",
        "EVENT": "🎉",
        "PRODUCT": "📦",
        "LANGUAGE": "🗣️",
        "NORP": "👥",
        "FAC": "🏗️",
        "LAW": "⚖️",
        "WORK_OF_ART": "🎨"
    }

    # 🔍 NLP Breakdown in Expander
    with st.expander("🔍 View Full NLP Breakdown"):
        st.markdown("**🔤 Tokens:**")
        st.write([f"🔹 {token.text}" for token in doc])

        st.markdown("**🧾 Lemmas:**")
        st.write([f"📄 {token.lemma_}" for token in doc])

        st.markdown("**📊 POS Tags:**")
        st.write([f"📌 {token.text} → {token.pos_}" for token in doc])

        # 🔄 Toggle for Entity View
        st.markdown("**🏷️ Named Entities:**")
        view_mode = st.radio("🔄 Choose entity view mode", ["🧾 Raw", "🏷️ Emoji-Mapped"])

        if doc.ents:
            if view_mode == "🧾 Raw":
                st.write([(ent.text, ent.label_) for ent in doc.ents])
            else:
                styled_ents = [
                    f"{ENTITY_EMOJI_MAP.get(ent.label_, '❓')} {ent.text} ({ent.label_})"
                    for ent in doc.ents
                ]
                st.write(styled_ents)
        else:
            st.info("ℹ️ No named entities found in the input.")

        # 🌥️ Wordcloud of Tokens
        st.markdown("**🌥️ Wordcloud of Tokens:**")
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        token_text = " ".join([token.text for token in doc])
        wc = WordCloud(width=800, height=400, background_color="white").generate(token_text)

        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # ☁️ Wordcloud of Lemmas
        st.markdown("**☁️ Wordcloud of Lemmas:**")
        lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        lemma_text = " ".join(lemmas)

        wc_lemma = WordCloud(width=600, height=300, background_color="white").generate(lemma_text)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc_lemma, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # 📊 POS Tag Distribution Chart
        import pandas as pd
        import plotly.express as px

        pos_counts = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

        pos_df = pd.DataFrame(list(pos_counts.items()), columns=["POS", "Count"])
        fig_pos = px.bar(pos_df, x="POS", y="Count", title="📊 POS Tag Distribution", color="POS")
        st.plotly_chart(fig_pos)

    st.stop()

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
DEFAULT_CSV_URL = "https://raw.githubusercontent.com/Vikrantthenge/sentiment-Analyzer/main/airline-reviews.csv"

uploaded_file = st.file_uploader("Upload airline-reviews.csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom file uploaded successfully.")
else:
    try:
        df = pd.read_csv(DEFAULT_CSV_URL)
        st.info("ℹ️ Using default demo file from GitHub")
    except Exception:
        st.error("❌ Default file not found. Please upload a CSV file.")
        st.stop()

st.write("📁 Active file:", uploaded_file.name if uploaded_file else "airline-reviews.csv")

# ✈️ Simulate airline column if missing
if "airline" not in df.columns:
    df["airline"] = [random.choice(["Indigo", "Air India", "SpiceJet", "Vistara", "Akasa", "Air Asia"]) for _ in range(len(df))]

# 🧠 Sentiment Analysis
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    huggingface_available = True
except Exception:
    st.warning("⚠️ Hugging Face model failed. Switching to VADER fallback...")
    huggingface_available = False
    analyzer = SentimentIntensityAnalyzer()

text_candidates = [col for col in df.columns if df[col].dtype == "object" and df[col].str.len().mean() > 30]
default_text_col = "text" if "text" in df.columns else (text_candidates[0] if text_candidates else None)

if default_text_col:
    st.markdown("### 📝 Select Text Column for Sentiment Analysis")
    selected_text_col = st.selectbox("Choose column containing customer feedback:", df.columns, index=df.columns.get_loc(default_text_col))

    try:
        if huggingface_available:
            df["sentiment"] = df[selected_text_col].apply(lambda x: sentiment_pipeline(str(x))[0]["label"].upper())
        else:
            def vader_sentiment(text):
                score = analyzer.polarity_scores(str(text))['compound']
                return "POSITIVE" if score >= 0.05 else "NEGATIVE" if score <= -0.05 else "NEUTRAL"
            df["sentiment"] = df[selected_text_col].apply(vader_sentiment)
    except Exception as e:
        st.error("❌ Error applying sentiment analysis. Please check if the selected column contains valid text.")
        st.exception(e)
        st.stop()
else:
    st.error("❌ No suitable text column found. Please upload a CSV with a column like 'text', 'review', or 'comments'.")
    st.stop()

# ✈️ Normalize airline names for dropdown visibility
df["airline"] = df["airline"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()

# 🔧 Manual mapping to standardize Akasa and AirAsia
airline_map = {
    "Air Asia": "AirAsia",
    "Air Asia India": "AirAsia",
    "Akasa Air": "Akasa",
    "Akasa Airlines": "Akasa"
}
df["airline"] = df["airline"].replace(airline_map)

# ✈️ Airline Filter
selected_airline = st.selectbox("✈️ Filter by Airline", sorted(df["airline"].unique()))
df = df[df["airline"] == selected_airline]

# 📊 Sentiment by Airline (Grouped Bar Chart)
st.markdown("### 📊 Sentiment Volume by Airline")
sentiment_airline = df.groupby(["airline", "sentiment"]).size().reset_index(name="count")
fig_grouped = px.bar(
    sentiment_airline,
    x="airline",
    y="count",
    color="sentiment",
    barmode="group",
    title="Sentiment Volume by Airline",
    color_discrete_map={"POSITIVE": "blue", "NEGATIVE": "red", "NEUTRAL": "gray"}
)
st.plotly_chart(fig_grouped, use_container_width=True)

# 📈 Sentiment Trend Over Time (Stacked Area)
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
        fig_trend = px.area(
            pivot_df,
            x="date",
            y="count",
            color="sentiment",
            title="Sentiment Over Time",
            color_discrete_map={"POSITIVE": "blue", "NEGATIVE": "crimson", "NEUTRAL": "gray"}
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # 📈 Rolling Average Sentiment Trend
        st.markdown("### 📈 Smoothed Sentiment Trend (7-Day Rolling Avg)")
        rolling_df = df.groupby(["date", "sentiment"]).size().unstack().fillna(0)
        rolling_avg = rolling_df.rolling(window=7).mean().reset_index().melt(id_vars="date", var_name="sentiment", value_name="count")
        fig_smooth = px.line(
            rolling_avg,
            x="date",
            y="count",
            color="sentiment",
            title="7-Day Rolling Average of Sentiment",
            color_discrete_map={"POSITIVE": "blue", "NEGATIVE": "red", "NEUTRAL": "gray"}
        )
        st.plotly_chart(fig_smooth, use_container_width=True)

        # 📍 Sentiment Heatmap by Weekday
        st.markdown("### 📍 Sentiment Heatmap by Weekday")
        df["weekday"] = df["date"].dt.day_name()
        heatmap_df = df.groupby(["weekday", "sentiment"]).size().unstack(fill_value=0)
        heatmap_df = heatmap_df.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        fig_heatmap = px.imshow(
            heatmap_df,
            labels=dict(x="Sentiment", y="Weekday", color="Count"),
            title="Sentiment Volume by Weekday",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # 📊 Diverging Sentiment Bar Chart
        st.markdown("### 📊 Diverging Sentiment by Date")
        div_df = df.groupby(["date", "sentiment"]).size().unstack(fill_value=0).reset_index()
        div_df["POSITIVE"] = div_df.get("POSITIVE", 0)
        div_df["NEGATIVE"] = -div_df.get("NEGATIVE", 0)
        div_melted = div_df.melt(id_vars="date", value_vars=["POSITIVE", "NEGATIVE"], var_name="sentiment", value_name="count")

        fig_diverge = px.bar(
            div_melted,
            y="date",
            x="count",
            color="sentiment",
            title="Diverging Sentiment by Date",
            color_discrete_map={"POSITIVE": "blue", "NEGATIVE": "red"},
            barmode="relative"
        )
        fig_diverge.update_layout(yaxis_title="Date", xaxis_title="Sentiment Count")
        st.plotly_chart(fig_diverge)

        # 🧭 Radar Chart — Airline Sentiment Profile
        st.markdown("### 🧭 Airline Sentiment Profile (Radar Chart)")
        radar_df = df.groupby(["airline", "sentiment"]).size().unstack(fill_value=0).reset_index()
        radar_df = radar_df.set_index("airline")
        fig_radar = px.line_polar(
            radar_df.reset_index().melt(id_vars="airline", var_name="sentiment", value_name="count"),
            r="count",
            theta="sentiment",
            color="airline",
            line_close=True,
            title="Sentiment Profile by Airline"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

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
        width=800, height=400, background_color='white',
        stopwords=custom_stopwords, collocations=False, max_words=100
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
    st.error(f"🚨 Spike in negative sentiment detected for {selected_airline}. Investigate recent feedback.")
else:
    st.success("✅ No major negative sentiment spike detected.")

# 📌 Footer Branding
st.markdown("---")
st.markdown("**✈️ From Runways to Regression Models — Aviation Expertise Meets Data Intelligence.**")
st.markdown("""
### 🔗 Connect with Me  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-vthenge-blue?logo=linkedin)](https://www.linkedin.com/in/vthenge)  
[![Email](https://img.shields.io/badge/Email-vikrantthenge@outlook.com-red?logo=gmail)](mailto:vikrantthenge@outlook.com)  
[![GitHub](https://img.shields.io/badge/GitHub-vikrantthenge-black?logo=github)](https://github.com/vikrantthenge)
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size:16px; font-weight:normal; color:#343a40; line-height:1.2;'>
🔍 Powered by NLP & CX Intelligence — Built for Airline Feedback Precision.<br>
🔐 This dashboard analyzes anonymized feedback only. No personal data is stored.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 16px; font-weight: bold; color: #000000;'>
🛠️ Version: v1.0 | 📅 Last Updated: October 2025
</div>
""", unsafe_allow_html=True)



                                 


