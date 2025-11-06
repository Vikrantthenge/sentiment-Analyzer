
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import nltk
import random
import re

# nltk.download("vader_lexicon", quiet=True) for streamlit

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Redirect NLTK download to a writable local directory
nltk.download("vader_lexicon", download_dir="./nltk_data", quiet=True)

# Tell NLTK to look in the local directory
nltk.data.path.append("./nltk_data")

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# 🌐 Page Config
st.set_page_config(page_title="✈️ Airline Sentiment Analyzer", layout="centered")

# 🖼️ Logo + Animated Header
# 🖼️ Logo + Animated Header with ✈️ Flight
# 🖼️ Logo + Animated Header with ✈️ Flight aligned to "Vikrant"
# 🖼️ Logo + Gradient Plane Animation + Typing Header
import base64
import streamlit as st

# Convert plane image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_base64 = get_base64_image("gradient_plane.png")

# Layout with logo + animated header block
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.png", width=100)

with col2:
    st.markdown(f"""
    <style>
    @keyframes typing {{
      from {{ width: 0 }}
      to {{ width: 100% }}
    }}

    @keyframes blink {{
      50% {{ border-color: transparent }}
    }}

    @keyframes fly {{
      0%   {{ left: -60px; opacity: 0 }}
      30%  {{ opacity: 1 }}
      100% {{ left: 420px; opacity: 0 }}
    }}

    .typing-header {{
      font-size: 32px;
      font-weight: bold;
      white-space: nowrap;
      overflow: hidden;
      border-right: 3px solid #0078D4;
      width: 0;
      animation: typing 3s steps(30, end) forwards, blink 0.75s step-end infinite;
      color: #0078D4;
      margin-bottom: 20px;
      position: relative;
    }}

    .flight-img {{
      position: absolute;
      animation: fly 4s linear infinite;
      top: -40px;
      width: 80px;
    }}
    </style>

    <div style="position: relative;">
      <img src="data:image/png;base64,{image_base64}" class="flight-img">
      <div class="typing-header">Airline Sentiment Analyzer by Vikrant</div>
    </div>
    """, unsafe_allow_html=True)


# 🔀 Mode Selection
mode = st.radio("Choose Mode", ["⚡ Basic Sentiment", "🧬 NLP Pipeline Demo"])

# 🤖 Load Hugging Face Pipeline
try:
    hf_pipeline = pipeline("sentiment-analysis")
    huggingface_available = True
except Exception:
    huggingface_available = False

# 🧠 VADER Fallback
vader = SentimentIntensityAnalyzer()

# ✨ Basic Sentiment Mode
if mode == "⚡ Basic Sentiment":
    st.markdown("## ⚡ Quick Sentiment Check")
    user_input = st.text_input("💬 Enter text for sentiment check", key="basic_input")

    if user_input.strip():
        try:
            sentiment = hf_pipeline(user_input)
            label = sentiment[0]["label"]
            score = round(sentiment[0]["score"], 3)
        except Exception:
            huggingface_available = False

        if huggingface_available:
            st.markdown("### 🤖 Hugging Face Sentiment")
            st.write({"Label": label, "Confidence Score": score})
        else:
            st.warning("⚠️ Hugging Face failed. Using VADER fallback.")
            sentiment_scores = vader.polarity_scores(user_input)
            compound = sentiment_scores["compound"]
            label = (
                "POSITIVE" if compound > 0.05 else
                "NEUTRAL" if -0.05 <= compound <= 0.05 else
                "NEGATIVE"
            )
            score = round(compound, 3)
            st.markdown("### 🛟 VADER Sentiment Fallback")
            st.write({"Label": label, "Compound Score": score})

        # 🎛️ Emoji Toggle
        display_mode = st.radio("🎛️ Display Mode", ["😊 Emoji View", "🔤 Plain Text View"])
        emoji_map = {
            "POSITIVE": "😊 Positive",
            "NEGATIVE": "😞 Negative",
            "NEUTRAL": "😐 Neutral"
        }
        final_label = emoji_map.get(label, "❓ Unknown") if display_mode == "😊 Emoji View" else label.capitalize()
        st.markdown(f"**Sentiment:** {final_label} ({score})")

# 🧬 NLP Pipeline Mode
elif mode == "🧬 NLP Pipeline Demo":
    st.markdown("## 🧬 NLP Pipeline Explorer")
    user_input = st.text_area("💬 Enter text for NLP processing", key="nlp_input")

    if user_input.strip():
        try:
            nlp = spacy.load("./en_core_web_sm/en_core_web_sm-3.8.0")
        except OSError:
            st.error("⚠️ spaCy model not found. Please bundle it correctly.")
            st.stop()

        # 📈 Sentiment Block
        try:
            sentiment = hf_pipeline(user_input)
            label = sentiment[0]["label"]
            score = round(sentiment[0]["score"], 3)
        except Exception:
            huggingface_available = False

        if huggingface_available:
            st.markdown("### 🤖 Hugging Face Sentiment")
            st.write({"Label": label, "Confidence Score": score})
        else:
            st.warning("⚠️ Hugging Face failed. Using VADER fallback.")
            sentiment_scores = vader.polarity_scores(user_input)
            compound = sentiment_scores["compound"]
            label = (
                "POSITIVE" if compound > 0.05 else
                "NEUTRAL" if -0.05 <= compound <= 0.05 else
                "NEGATIVE"
            )
            score = round(compound, 3)
            st.markdown("### 🛟 VADER Sentiment Fallback")
            st.write({"Label": label, "Compound Score": score})

        display_mode = st.radio("🎛️ Display Mode", ["😊 Emoji View", "🔤 Plain Text View"])
        emoji_map = {
            "POSITIVE": "😊 Positive",
            "NEGATIVE": "😞 Negative",
            "NEUTRAL": "😐 Neutral"
        }
        final_label = emoji_map.get(label, "❓ Unknown") if display_mode == "😊 Emoji View" else label.capitalize()
        st.markdown(f"**Sentiment:** {final_label} ({score})")

        # 🔍 NLP Breakdown
        doc = nlp(user_input)
        ENTITY_EMOJI_MAP = {
            "PERSON": "🧑", "ORG": "🏢", "GPE": "🌍", "LOC": "📍", "DATE": "📅",
            "TIME": "⏰", "MONEY": "💰", "QUANTITY": "🔢", "EVENT": "🎉", "PRODUCT": "📦",
            "LANGUAGE": "🗣️", "NORP": "👥", "FAC": "🏗️", "LAW": "⚖️", "WORK_OF_ART": "🎨"
        }

        with st.expander("🔍 View Full NLP Breakdown"):
            st.markdown("**🔤 Tokens:**")
            st.write([f"🔹 {token.text}" for token in doc])

            st.markdown("**🧾 Lemmas:**")
            st.write([f"📄 {token.lemma_}" for token in doc])

            st.markdown("**📊 POS Tags:**")
            st.write([f"📌 {token.text} → {token.pos_}" for token in doc])

            st.markdown("**🏷️ Named Entities:**")
            view_mode = st.radio("🔄 Entity View", ["🧾 Raw", "🏷️ Emoji-Mapped"])
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
                st.info("ℹ️ No named entities found.")

            # 🌥️ Wordclouds
            token_text = " ".join([token.text for token in doc])
            wc = WordCloud(width=800, height=400, background_color="white").generate(token_text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
            lemma_text = " ".join(lemmas)
            wc_lemma = WordCloud(width=600, height=300, background_color="white").generate(lemma_text)
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wc_lemma, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)

            # 📊 POS Distribution
            pos_counts = {}
            for token in doc:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
            pos_df = pd.DataFrame(list(pos_counts.items()), columns=["POS", "Count"])
            fig_pos = px.bar(pos_df, x="POS", y="Count", title="📊 POS Tag Distribution", color="POS")
            st.plotly_chart(fig_pos)

    else:
        st.info("ℹ️ Please enter text to run the NLP pipeline.")


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

        # 🎯 NPS Scoring + Trendline
if "NPS" in df.columns:
    st.markdown("### 🎯 NPS Score Trend")
    df["NPS_Category"] = df["NPS"].apply(lambda x: "Promoter" if x >= 9 else "Passive" if x >= 7 else "Detractor")
    promoters = (df["NPS_Category"] == "Promoter").sum()
    detractors = (df["NPS_Category"] == "Detractor").sum()
    total_responses = len(df)
    nps_score = ((promoters - detractors) / total_responses) * 100
    st.metric("Net Promoter Score (NPS)", f"{nps_score:.1f}")

    if "date" in df.columns:
        nps_trend = df.groupby("date")["NPS"].mean().reset_index()
        fig_nps = px.line(nps_trend, x="date", y="NPS", title="📈 Average NPS Over Time")
        st.plotly_chart(fig_nps, use_container_width=True)


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

# 🧩 Theme Clustering using BERTopic
st.markdown("### 🧩 Theme Clustering (BERTopic)")

try:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    sample_texts = df[selected_text_col].dropna().astype(str).tolist()
    if len(sample_texts) > 30:
        st.info("🧠 Generating topics using BERTopic (may take a moment)...")
        topic_model = BERTopic(verbose=False)
        topics, _ = topic_model.fit_transform(sample_texts)
        topic_freq = topic_model.get_topic_info()
        st.dataframe(topic_freq.head(10))
        fig_topics = topic_model.visualize_barchart(top_n_topics=5)
        st.plotly_chart(fig_topics, use_container_width=True)
    else:
        st.warning("Not enough text data to run BERTopic clustering.")
except Exception as e:
    st.warning(f"⚠️ BERTopic unavailable or dependency issue: {e}")

# 🧭 CX Journey Segmentation
st.markdown("### 🧭 CX Journey Segmentation")

if "date" in df.columns and "sentiment" in df.columns:
    df["journey_phase"] = pd.cut(df["date"].rank(method="first"),
                                 bins=3, labels=["Pre-Flight", "In-Flight", "Post-Flight"])
    phase_sentiment = df.groupby(["journey_phase", "sentiment"]).size().reset_index(name="count")
    fig_phase = px.bar(phase_sentiment, x="journey_phase", y="count", color="sentiment",
                       title="CX Journey Sentiment Segmentation", barmode="group")
    st.plotly_chart(fig_phase, use_container_width=True)
else:
    st.info("Add a 'date' column to enable CX journey segmentation.")


# ⚠️ CX Alert Section
st.markdown("### ⚠️ CX Alert")
neg_count = sentiment_counts.get("NEGATIVE", 0)
if neg_count > 10:
    st.error(f"🚨 Spike in negative sentiment detected for {selected_airline}. Investigate recent feedback.")
else:
    st.success("✅ No major negative sentiment spike detected.")

# 📤 Export to PDF Summary Report
st.markdown("### 📤 Export Summary Report")

from fpdf import FPDF

if st.button("Generate PDF Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Airline Sentiment Summary Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"Selected Airline: {selected_airline}")
    pdf.multi_cell(0, 10, txt=f"Total Records: {len(df)}")
    if "NPS" in df.columns:
        pdf.multi_cell(0, 10, txt=f"NPS Score: {nps_score:.1f}")
    pdf.multi_cell(0, 10, txt="Top 5 Topics / Themes:")
    if "topic_model" in locals():
        top_topics = topic_model.get_topic_info().head(5).to_string(index=False)
        pdf.multi_cell(0, 10, txt=top_topics)
    pdf.output("CX_Summary_Report.pdf")
    st.success("✅ PDF report generated successfully! Check your project folder.")


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



# ===========================================================
# 💠 QUALTRICS COMPATIBILITY SECTION
# ===========================================================

st.markdown("### 💠 Qualtrics-Compatible Data Import")

def load_qualtrics_csv(file):
    try:
        # Try reading normally first
        df = pd.read_csv(file)
        # Detect Qualtrics format (metadata rows start with 'StartDate' in row 1)
        if df.columns[0].startswith("StartDate") or "ResponseId" in df.columns:
            # Re-read, skipping first two metadata rows (common in Qualtrics exports)
            file.seek(0)
            df = pd.read_csv(file, skiprows=[0, 1])
            st.info("💡 Detected Qualtrics survey format. Automatically cleaned headers.")
        return df
    except Exception as e:
        st.error(f"❌ Could not read Qualtrics CSV: {e}")
        return None
    
# 🔄 Simulated Qualtrics Pull Toggle
st.markdown("### 🔄 Simulated Qualtrics Pull")

simulate_pull = st.toggle("Simulate Real-Time Qualtrics Pull")
if simulate_pull:
    st.info("🔄 Simulating Qualtrics API pull... (demo data refreshed)")
    # Example: Add random NPS responses or timestamps to mimic API feed
    df["Timestamp"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df))
    df["NPS"] = [random.randint(0, 10) for _ in range(len(df))]
    st.success("✅ Simulated pull complete with timestamps and NPS field added.")


# Replace standard upload handling
DEFAULT_CSV_URL = "https://raw.githubusercontent.com/Vikrantthenge/sentiment-Analyzer/main/airline-reviews.csv"

uploaded_file = st.file_uploader("Upload airline-reviews.csv or Qualtrics export", type=["csv"])
if uploaded_file is not None:
    df = load_qualtrics_csv(uploaded_file)
    if df is not None:
        st.success("✅ File uploaded successfully (Qualtrics-compatible).")
    else:
        st.stop()
else:
    try:
        df = pd.read_csv(DEFAULT_CSV_URL)
        st.info("ℹ️ Using default demo file from GitHub")
    except Exception:
        st.error("❌ Default file not found. Please upload a CSV file.")
        st.stop()

# Add badge in sidebar
with st.sidebar:
    st.markdown("### 💠 Qualtrics-Compatible ✅")
    st.markdown("This app can process **Qualtrics survey exports** directly (auto-detects and cleans headers).")

# ===========================================================
