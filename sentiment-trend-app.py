import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from transformers import pipeline

# 🌐 Page Config
st.set_page_config(page_title="📊 Sentiment Trend Analyzer", layout="centered")

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
    st.markdown("<div class='typing-header'>Sentiment Trend Analyzer by Vikrant</div>", unsafe_allow_html=True)

# 📘 About Section
st.markdown("<div class='glow-line'></div>", unsafe_allow_html=True)
st.markdown("### About This App")
st.markdown("""
Welcome to the **Sentiment Trend Analyzer** — A NLP demo built to showcase real-time sentiment analysis using Hugging Face transformers.

This app allows you to:
- 📊 Analyze sentiment from any uploaded CSV text column
- 📈 Visualize confidence trends over time
- 🧠 Explore sentiment distribution with intuitive charts
- 📥 Download annotated results instantly

Built by **Vikrant Thenge**, a data-driven problem solver with expertise in predictive modeling, dashboard development, and cloud-native deployment.
""")

# 📤 Upload Section
st.markdown("<div class='glow-line'></div>", unsafe_allow_html=True)
st.markdown("### 📤 Upload Your CSV")
st.info("📱 Tip: For best experience on mobile, use Chrome or Safari. Some in-app browsers may not support file uploads properly.")
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

# 📎 Sample CSV Fallback
sample_data = {
    "date": ["2025-08-01", "2025-08-02", "2025-08-03"],
    "text": ["Great product!", "Terrible service.", "Average experience."]
}
sample_df = pd.DataFrame(sample_data)
sample_csv = sample_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Sample CSV", sample_csv, "sample_sentiment.csv", "text/csv")

with st.expander("📘 Sample CSV Format Guide"):
    st.markdown("""
Your CSV should include the following columns:
- `date`: Date of the comment or review
- `text`: The actual feedback or message

Example:

                """)

# 📦 Load Data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.toast("File uploaded!")  # Mobile feedback
else:
    df = sample_df
    st.info("📎 No file uploaded. Using sample data.")

# 🧹 Normalize Columns
df.columns = [col.strip().lower() for col in df.columns]
required_columns = {"date", "text"}
if not required_columns.issubset(df.columns):
    st.error("❌ CSV must contain 'date' and 'text' columns.")
    st.stop()

# 🔍 Column Selection
st.markdown("### 🔍 Select Columns")
text_col = st.selectbox("Text Column", df.columns, index=df.columns.get_loc("text"))
date_col = st.selectbox("Date Column", df.columns, index=df.columns.get_loc("date"))

# 🤖 Sentiment Analysis
st.markdown("### 🧠 Sentiment Analysis Results")
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
sentiment_pipeline = load_model()

if "sentiment" not in df.columns or "confidence" not in df.columns:
    df["sentiment"] = df[text_col].apply(lambda x: sentiment_pipeline(str(x))[0]['label'])
    df["confidence"] = df[text_col].apply(lambda x: sentiment_pipeline(str(x))[0]['score'])

df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
st.dataframe(df[[date_col, text_col, "sentiment", "confidence"]], use_container_width=True)

# 📈 Confidence Trend
st.markdown("### 📈 Confidence Over Time")
fig, ax = plt.subplots()
ax.plot(df[date_col], df["confidence"], marker='o', linestyle='-', color='teal')
ax.set_xlabel("Date")
ax.set_ylabel("Confidence Score")
ax.set_title("Sentiment Confidence Over Time")
interval = max(1, df[date_col].nunique() // 6)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig.autofmt_xdate()
st.pyplot(fig, use_container_width=True)

# 📊 Sentiment Distribution
st.markdown("### 📊 Sentiment Distribution")
sentiment_counts = df["sentiment"].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99','#ffcc99'])
ax2.axis('equal')
st.pyplot(fig2, use_container_width=True)

# 📥 Download Results
st.markdown("### 📥 Download Annotated CSV")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "sentiment_results.csv", "text/csv")

# 🧾 Footer Branding
st.markdown("<div class='glow-line'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 14px;'>
NLP Demo Made with ❤️ by <b>Vikrant Thenge</b>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div class='contact-strip'>
<a href='mailto:vikrantthenge@outlook.com'>
<img src='https://cdn-icons-png.flaticon.com/512/732/732223.png' width='20'> vikrantthenge@outlook.com
</a>
<a href='https://github.com/vikrantthenge' target='_blank'>
<img src='https://cdn-icons-png.flaticon.com/512/25/25231.png' width='20'> GitHub
</a>
<a href='https://www.linkedin.com/in/vikrantthenge/' target='_blank'>
<img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='20'> LinkedIn
</a>
</div>
""", unsafe_allow_html=True)
