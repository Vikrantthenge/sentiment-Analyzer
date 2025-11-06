## ✈️ Airline Sentiment Analyzer

**Built by Vikrant Thenge — NLP Strategist & Dashboard Architect**  
[![Location](https://img.shields.io/badge/Mumbai-based-6c757d?logo=googlemaps)](https://www.google.com/maps/place/Mumbai)  
[![Recruiter](https://img.shields.io/badge/Recruiter-Facing-0078D4?logo=target)](https://www.linkedin.com/in/vthenge)  
[![CX](https://img.shields.io/badge/CX-Intelligence-00C853?logo=insights)](https://sentiment-analyzer-vikrant.streamlit.app)
[![Survey](https://img.shields.io/badge/Qualtrics-Integrated-darkred?logo=qualtrics)](https://sentiment-analyzer-vikrant.streamlit.app)


[![LinkedIn](https://img.shields.io/badge/LinkedIn-vthenge-blue?logo=linkedin)](https://www.linkedin.com/in/vthenge)  
[![GitHub](https://img.shields.io/badge/GitHub-vikrantthenge-black?logo=github)](https://github.com/vikrantthenge)  
[![CI Status](https://github.com/Vikrantthenge/sentiment-Analyzer/actions/workflows/sentiment.ci.yml/badge.svg)](https://github.com/Vikrantthenge/sentiment-Analyzer/actions/workflows/sentiment.ci.yml)

---

### 🚀 Launch & Capabilities

<p align="center">
  <a href="https://sentiment-analyzer-vikrant.streamlit.app">
    <img src="https://img.shields.io/badge/Streamlit%20App-Live-green?style=for-the-badge&logo=streamlit">
  </a>
  <a href="https://huggingface.co/spaces/vthenge/sentiment-analyzer">
    <img src="https://img.shields.io/badge/Hugging%20Face-Live-orange?style=for-the-badge&logo=huggingface">
  </a>
  <img src="https://img.shields.io/badge/NLP%20Pipeline-Enabled-blue?style=for-the-badge&logo=spacy">
  <img src="https://img.shields.io/badge/Visual%20Insights-WordCloud%20%26%20POS-orange?style=for-the-badge&logo=plotly">
</p>

---

## 🧠 Overview

A recruiter-facing NLP dashboard built with Streamlit to analyze airline customer feedback.  
It applies sentiment analysis to passenger reviews and visualizes trends for CX teams, route planners, and service recovery leads.
Compatible with CSV uploads and Qualtrics survey exports, enabling recruiter and passenger feedback loops.

---

## 🚀 Features

✅ **Sentiment Analysis** using HuggingFace Transformers  
📈 **Trendline** with red/blue polarity mapping  
📊 **Diverging Bar Chart** for sentiment swings  
🧠 **Word Cloud** for frequent negative keywords  
✈️ **Airline Filter** to isolate carrier-specific feedback  
⚠️ **CX Alert System** for spikes in negative sentiment  
📁 **CSV Upload** or use the default demo file  
🎨 **Animated branding**, sidebar polish, and footer badges  
🧬 **NLP Pipeline Demo** with tokenization, lemmatization, entity recognition  
☁️ **Word Cloud of Lemmas** for theme discovery  
📊 **POS Tag Distribution** chart for linguistic breakdown
🧾 Qualtrics-Compatible for survey-based sentiment analysis and recruiter feedback integration


---

## 📦 Tech Stack

🧩 **Streamlit** for UI  
🤖 **Transformers (DistilBERT)** for sentiment analysis  
📊 **Plotly** for interactive charts  
🖼️ **Matplotlib & WordCloud** for visual NLP  
📐 **Pandas** for data wrangling  
🧬 **spaCy** for NLP pipeline tasks  
🔄 **GitHub Actions** for CI/CD automation

![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Transformers](https://img.shields.io/badge/HuggingFace-DistilBERT-yellow?logo=huggingface)
![Plotly](https://img.shields.io/badge/Plotly-Charts-orange?logo=plotly)
![spaCy](https://img.shields.io/badge/spaCy-NLP-blue?logo=spacy)
![WordCloud](https://img.shields.io/badge/WordCloud-Visuals-lightgrey?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Wrangling-black?logo=pandas)
![Qualtrics](https://img.shields.io/badge/Qualtrics-Surveys-darkred?style=flat-square&logo=qualtrics)
![Google Forms](https://img.shields.io/badge/Google%20Forms-Feedback-purple?style=flat-square&logo=googleforms)
![Typeform](https://img.shields.io/badge/Typeform-UX%20Surveys-black?style=flat-square&logo=typeform)

## 🔹 Survey Compatibility  
📊 Survey Tools: Qualtrics | Google Forms | Typeform

---

## 🔄 CI/CD Integration

This project uses **GitHub Actions** for Continuous Integration:

- ✅ Lint checks via `flake8`  
- ✅ Optional unit tests via `pytest`  
- ✅ Auto-deployment via Streamlit Cloud

Every push to `main` triggers automated validation and deployment, ensuring clean, reliable code with zero manual effort.

---

## ✨ Sample NLP Output

**Input:**  
`The flight from Mumbai to Delhi had no entertainment and poor service.`

**Output:**  
- 🔤 Tokens: `["The", "flight", "from", "Mumbai", "to", "Delhi", "had", "no", "entertainment", "and", "poor", "service"]`  
- 🧾 Lemmas: `["the", "flight", "from", "Mumbai", "to", "Delhi", "have", "no", "entertainment", "and", "poor", "service"]`  
- 🏷️ Named Entities: `Mumbai (GPE), Delhi (GPE)`  
- 📊 POS Tags: `NOUN, PROPN, VERB, ADJ...`  
- ☁️ Word Cloud: Highlights `"entertainment"`, `"service"`, `"poor"`  
- 📊 POS Chart: Shows noun/verb/adjective distribution

---

## ✈️ Airline NLP Pipeline Demo

🔍 Explore named entity recognition with emoji-mapped clarity — tuned for airline apps and passenger data.

| Feature | Description |
|--------|-------------|
| 🧬 NLP Pipeline | Tokenization, Lemmatization, POS tagging |
| 🏷️ Entity Mapping | Auto-decorated with emojis for PERSON, ORG, GPE, DATE, MONEY, TIME |
| 🌥️ Wordclouds | Token and Lemma-based visual summaries |
| 📊 POS Chart | Interactive bar chart via Plotly |
| 🔄 Entity Toggle | Switch between Raw and Emoji-Mapped views |

🚀 Sample Input:  
> John booked a flight with Indigo Airlines from Mumbai to Dubai on October 15th, 2025. He paid ₹32,000 and requested a vegetarian meal. The flight departs at 9:30 AM and arrives at 12:45 PM local time.

🧑 John (PERSON)  
🏢 Indigo Airlines (ORG)  
🌍 Mumbai → Dubai (GPE)  
📅 October 15th, 2025 (DATE)  
💰 ₹32,000 (MONEY)  
⏰ 9:30 AM → 12:45 PM (TIME)

---

### 🔗 Launch Demo

<p align="center">
  <a href="https://sentiment-analyzer-vikrant.streamlit.app">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
  </a>
  &nbsp;&nbsp;
  <a href="https://huggingface.co/spaces/vthenge/sentiment-analyzer">
    <img src="https://img.shields.io/badge/Open%20in-Hugging%20Face-orange?logo=huggingface&style=flat-square" alt="Open in Hugging Face">
  </a>
</p>

📦 Powered by `spaCy`, `Streamlit`, `WordCloud`, `Plotly`  
🧠 Branded by VT | Built for recruiter clarity

---

## 📄 Sample CSV Format

```csv
date,airline,text
01-06-2025,Indigo,Amazing experience! Flight was on time and crew was very friendly.
02-06-2025,Air India,Worst flight ever. Extremely delayed and no communication.
...
