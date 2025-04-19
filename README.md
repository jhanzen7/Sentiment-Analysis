# Sentiment and Emotional Analysis Web App

This project is an interactive web application for performing **sentiment** and **emotion classification** on user-provided text or product reviews. It leverages state-of-the-art transformer models and intuitive visualizations to break down emotional tone and highlight sentiment-rich phrases.

## Features

- Overall sentiment classification using `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Emotion classification with `monologg/bert-base-cased-goemotions-original`
- Sentence-level breakdown with individual sentiment and emotion detection
- Highlighting sentiment-heavy chunks within text
- Bar graph visualizations using Plotly
- Random Amazon review generator (from a dataset)
- 
## Requirements

Ensure you have **Python 3.8+** and install the following dependencies:

```bash
pip install -r requirements.txt
```

You also need to download the following language model for spaCy:

```bash
python -m spacy download en_core_web_sm
```

## Running the App

Launch the Streamlit application with:

```bash
streamlit run sentiment_analysis_app.py
```

Then open the app in your browser at:

```
http://localhost:8501
```

© 2025 Jonah Hanzen — Powered by HuggingFace and Streamlit.
