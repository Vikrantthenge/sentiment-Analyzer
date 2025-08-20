import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from transformers import pipeline


# 📄 Page config
st.set_page_config(page_title="Sentiment Trend Analyzer", layout="wide")

# 📢 Instructions
st.markdown("""
### 📊 Sentiment Trend Analyzer
Upload your own CSV file or use the sample dataset provided.

**Required CSV Format:**
- `date`: Date of the entry (e.g., 2023-08-20)
- `text`: Text content to analyze

👉 You can also [download the sample CSV](https://github.com/Vikrantthenge/sentiment-Analyzer/blob/main/sentiment_sample.csv) to see the format.
""")

# 📁 File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# 📦 Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("sentiment_sample.csv")  # Make sure this file is in your repo

# ✅ Validate columns
required_columns = {"date", "text"}
if not required_columns.issubset(df.columns):
    st.error("❌ CSV must contain 'date' and 'text' columns.")
    st.stop()

# 👇 Proceed with sentiment analysis below this line
# Example: st.write(df.head())

# 🧠 Page Config with Logo + Favicon
st.set_page_config(
    page_title="Sentiment Trend Analyzer",
    page_icon="favicon.ico",
    layout="centered"
)

# 🎨 Branding Styles: Animated Header + Glowing Divider + Contact Strip
st.markdown("""
<style>
/* Animated Header */
@keyframes typing {
  from { width: 0 }
  to { width: 100% }
}
@keyframes blink {
  50% { border-color: transparent }
}
.typing-header {
  font-size: 32px;
  font-weight: bold;
  white-space: nowrap;
  overflow: hidden;
  border-right: 3px solid #0078D4;
  width: 0;
  animation: typing 3s steps(30, end) forwards, blink 0.75s step-end infinite;
  color: #0078D4;
  margin-bottom: 20px;
}

/* Glowing Divider */
.glow-line {
  height: 2px;
  background: linear-gradient(90deg, #0078D4, #00B4FF);
  animation: glow 2s infinite alternate;
  margin: 20px 0;
}
@keyframes glow {
  from { opacity: 0.4 }
  to { opacity: 1 }
}

/* Contact Strip */
.contact-strip {
  display: flex;
  align-items: center;
  gap: 20px;
  font-size: 16px;
  margin-top: 10px;
}
.contact-strip a {
  text-decoration: none;
  color: #333;
}
.contact-strip img {
  vertical-align: middle;
  margin-right: 5px;
}
</style>
""", unsafe_allow_html=True)

# 🖼️ Logo + Animated Header Block
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.png", width=80)
with col2:
    st.markdown("<div class='typing-header'>Sentiment Trend Analyzer</div>", unsafe_allow_html=True)

# 📘 About Section
st.markdown("<div class='glow-line'></div>", unsafe_allow_html=True)
st.markdown("### 📘 About This App")
st.markdown("""
Welcome to the **Sentiment Trend Analyzer** — a recruiter-ready NLP demo built to showcase real-time sentiment analysis using Hugging Face transformers.

This app allows you to:
- 🔍 Analyze sentiment from any uploaded CSV text column
- 📈 Visualize confidence trends over time
- 🧠 Explore sentiment distribution with intuitive charts
- 📥 Download annotated results instantly

Built by **Vikrant Thenge**, a data-driven problem solver with expertise in predictive modeling, dashboard development, and cloud-native deployment.
""")

# 📤 Upload Section
st.markdown("<div class='glow-line'></div>", unsafe_allow_html=True)
st.markdown("### 📤 Upload Your CSV")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 🔧 Column Selection
    st.markdown("### 🔧 Select Columns")
    text_col = st.selectbox("Text Column", df.columns)
    date_col = st.selectbox("Date Column", df.columns)

    # 🧠 Sentiment Analysis
    st.markdown("### 📊 Sentiment Analysis Results")

    @st.cache_resource
    def load_model():
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    sentiment_pipeline = load_model()
    df["Sentiment"] = df[text_col].apply(lambda x: sentiment_pipeline(str(x))[0]['label'])
    df["Confidence"] = df[text_col].apply(lambda x: sentiment_pipeline(str(x))[0]['score'])
    df[date_col] = pd.to_datetime(df[date_col])

    st.dataframe(df[[date_col, text_col, "Sentiment", "Confidence"]])

    # 📈 Confidence Trend
    st.markdown("### 📈 Confidence Over Time")
    fig, ax = plt.subplots()
    ax.plot(df[date_col], df["Confidence"], marker='o', linestyle='-', color='teal')
    ax.set_xlabel("Date")
    ax.set_ylabel("Confidence Score")
    ax.set_title("Sentiment Confidence Over Time")
    interval = max(1, df[date_col].nunique() // 6)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate()
    st.pyplot(fig)

    # 📊 Sentiment Distribution
    st.markdown("### 📊 Sentiment Distribution")
    sentiment_counts = df["Sentiment"].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99'])
    ax2.axis('equal')
    st.pyplot(fig2)

    # 📥 Download Results
    st.markdown("### 📥 Download Annotated CSV")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "sentiment_results.csv", "text/csv")

else:
    st.info("Please upload a CSV file to begin.")

# 📌 Footer Branding
st.markdown("<div class='glow-line'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 14px;'>
Made with 🤖 by <b>Vikrant Thenge</b> | Recruiter-Ready NLP Demo
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
