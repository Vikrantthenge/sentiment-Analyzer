# ✈️ Airline Sentiment Analyzer

[![Live App](https://img.shields.io/badge/Streamlit-Live_App-00C853?logo=streamlit)](https://sentiment-analyzer-vikrant.streamlit.app)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-vthenge-blue?logo=linkedin)](https://www.linkedin.com/in/vthenge)
[![GitHub](https://img.shields.io/badge/GitHub-vikrantthenge-black?logo=github)](https://github.com/vikrantthenge)
[![Sentiment Analyzer CI](https://github.com/Vikrantthenge/sentiment-Analyzer/actions/workflows/sentiment.ci.yml/badge.svg)](https://github.com/Vikrantthenge/sentiment-Analyzer/actions/workflows/sentiment.ci.yml)

---

## 🧠 Overview

A recruiter-facing NLP dashboard built with Streamlit to analyze airline customer feedback.  
It applies sentiment analysis to passenger reviews and visualizes trends for CX teams, route planners, and service recovery leads.

---

## 🚀 Features

- ✅ Sentiment Analysis using HuggingFace Transformers  
- 📈 Trendline with red/blue polarity mapping  
- 📊 Diverging Bar Chart for sentiment swings  
- 🧠 Word Cloud for frequent negative keywords  
- ✈️ Airline Filter to isolate carrier-specific feedback  
- ⚠️ CX Alert System for spikes in negative sentiment  
- 📁 Upload your own CSV or use the default demo file  
- 🎨 Animated branding, sidebar polish, and footer badges

---

## 📦 Tech Stack

- **Streamlit** for UI  
- **Transformers (DistilBERT)** for sentiment analysis  
- **Plotly** for interactive charts  
- **Matplotlib & WordCloud** for visual NLP  
- **Pandas** for data wrangling  
- **GitHub Actions** for CI/CD automation

---

## 🔄 CI/CD Integration

This project uses **GitHub Actions** for Continuous Integration:

- ✅ Lint checks via `flake8`
- ✅ Optional unit tests via `pytest`
- ✅ Auto-deployment via Streamlit Cloud

Every push to `main` triggers automated validation and deployment, ensuring clean, reliable code with zero manual effort.

---

## 📄 Sample CSV Format

```csv
date,airline,text
01-06-2025,Indigo,Amazing experience! Flight was on time and crew was very friendly.
02-06-2025,Air India,Worst flight ever. Extremely delayed and no communication.
...

