# 📈 Sentiment Trend Over Time (Smoothed for clarity)
st.markdown("### 📈 Sentiment Over Time")

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
            title="📈 Sentiment Over Time",
            line_shape="spline",
            color_discrete_map={"POSITIVE": "blue", "NEGATIVE": "crimson"},
            template="plotly_white"
        )
        fig_trend.update_layout(hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No date column found. Trendline skipped.")

# 📊 Stacked Sentiment Bar Chart
st.markdown("### 📊 Sentiment Breakdown by Date")

div_df = df_trend.groupby(["date", "sentiment"]).size().reset_index(name="count")
fig_stack = px.bar(
    div_df,
    x="date",
    y="count",
    color="sentiment",
    title="📊 Sentiment Breakdown by Date",
    color_discrete_map={"POSITIVE": "blue", "NEGATIVE": "red"},
    barmode="stack",
    template="plotly_white"
)
fig_stack.update_layout(yaxis_title="Sentiment Count", xaxis_title="Date")
st.plotly_chart(fig_stack, use_container_width=True)

# 📊 Sentiment Distribution (Interactive Pie Chart)
st.markdown("### 📊 Overall Sentiment Distribution")

sentiment_counts = df["sentiment"].value_counts()
fig_pie = px.pie(
    names=sentiment_counts.index,
    values=sentiment_counts.values,
    title="📊 Sentiment Distribution",
    color_discrete_map={"POSITIVE": "blue", "NEGATIVE": "red"},
    template="plotly_white"
)
st.plotly_chart(fig_pie, use_container_width=True)

# 🧠 Word Cloud Toggle for Frequent Keywords
st.markdown("### 🧠 Frequent Keywords by Sentiment")

sentiment_focus = st.radio("Choose sentiment for word cloud:", ["NEGATIVE", "POSITIVE"])

custom_stopwords = set(STOPWORDS)
custom_stopwords.update([
    "positive", "negative", "neutral",
    "POSITIVE", "NEGATIVE", "NEUTRAL",
    "experience", "service", "flight", "airline",
    "good", "bad", "okay", "delay", "delayed", "late", "on", "off", "get", "got"
])

text_series = df[df["sentiment"] == sentiment_focus][selected_text_col].dropna().astype(str)

tokens = []
for text in text_series:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered = [word for word in words if word not in custom_stopwords]
    tokens.extend(filtered)

word_text = " ".join(tokens)

if word_text.strip():
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=custom_stopwords,
        collocations=False,
        max_words=100
    ).generate(word_text)

    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc, use_container_width=True)
else:
    st.info(f"No {sentiment_focus.lower()} sentiment found for this airline.")


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

