import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
from nltk.tokenize import sent_tokenize
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
import torch
import pandas as pd
import random

nlp = spacy.load('en_core_web_sm')
AMAZON_PATH = 'Dataset/Amazon.csv'

# Load the models
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
emotion_model = AutoModelForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
                      'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
                      'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
                      'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 
                      'remorse', 'sadness', 'surprise', 'neutral']

emotion_labels_with_emojis = {
    'admiration': '👏', 'amusement': '😂', 'anger': '😡', 'annoyance': '😒',
    'approval': '👍', 'caring': '🤗', 'confusion': '😕', 'curiosity': '🤔',
    'desire': '😍', 'disappointment': '😞', 'disapproval': '👎', 'disgust': '🤢',
    'embarrassment': '😳', 'excitement': '🤩', 'fear': '😨', 'gratitude': '🙏',
    'grief': '😭', 'joy': '😊', 'love': '❤️', 'nervousness': '😬', 'optimism': '🤞',
    'pride': '😌', 'realization': '💡', 'relief': '😌', 'remorse': '😔', 'sadness': '😢',
    'surprise': '😲', 'neutral': '😐'
}

def get_sentiment(text):
    result = sentiment_pipeline(text)[0]  # Get the first result from the list
    label = result['label']  # This will be 'POSITIVE' or 'NEGATIVE'
    score = result['score']  # Confidence score

    if label == 'positive':
        return 'Positive', score
    else:
        return 'Negative', score
    
def analyze_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = emotion_model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)
    
    # Get top emotion and confidence score
    sorted_emotions = sorted(
            [{"label": emotion_labels[idx], "score": predictions[0, idx].item()} for idx in range(len(predictions[0]))],
            key=lambda x: x["score"], reverse=True
        )
        
        # Add corresponding emojis for each emotion label
    for emotion in sorted_emotions:
        emotion['emoji'] = emotion_labels_with_emojis.get(emotion['label'], '')  # Get emoji for the label
        
    return sorted_emotions

def get_sentiment_chunks(text, sentiment_pipeline, threshold=0.5):
    doc = nlp(text)
    sentiment_chunks = []
    
    # Extract noun phrases (chunks) from the text
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text
        # Perform sentiment analysis on the whole chunk
        result = sentiment_pipeline(chunk_text)[0]
        score = result['score']
        label = result['label']
        print(f"chunk={chunk_text}, score={score}, label={label}")
        
        # Only keep the chunks that have strong sentiment (above the threshold)
        if score > threshold:
            sentiment_chunks.append((chunk_text, label, score))
    
    return sentiment_chunks

def split_into_chunks(text):
    # Use regex to split on commas, semicolons, conjunctions, and sentence delimiters
    split_pattern = r'[.,;!?]|(?:\bbut\b|\band\b|\bor\b|\bso\b)'
    chunks = re.split(split_pattern, text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]  # Clean up whitespace
    return chunks


def highlight_sentiment_chunks(text, sentiment_pipeline, threshold=0.5):
    chunks = split_into_chunks(text)
    highlighted_text = text

    for chunk in chunks:
        # Perform sentiment analysis on the chunk
        result = sentiment_pipeline(chunk)[0]
        label = result['label']
        score = result['score']

        # Highlight if score is above a certain threshold
        if label == 'positive' and score > threshold:
            highlighted_text = highlighted_text.replace(chunk, f"<span style='background-color: lightgreen'>{chunk}</span>")
        elif label == 'negative' and score > threshold:
            highlighted_text = highlighted_text.replace(chunk, f"<span style='background-color: lightcoral'>{chunk}</span>")

    return highlighted_text


def analyze_sentences(text, sentiment_pipeline):
    sentences = sent_tokenize(text, language='english')
    
    sentences = [sentence for sentence in sentences if re.search(r'\w+', sentence)]  # Keep only sentences with words

    sentence_analysis = []
    for sentence in sentences:
        result = sentiment_pipeline(sentence)[0]
        sentence_analysis.append((sentence, result['label'], result['score']))
    return sentence_analysis

def analyze_sentence_emotions(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = emotion_model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)
    
    # Get top emotion and confidence score
    sorted_emotions = sorted(
        [{"label": emotion_labels[idx], "score": predictions[0, idx].item()} for idx in range(len(predictions[0]))],
        key=lambda x: x["score"], reverse=True
    )
    
    # Add corresponding emojis for each emotion label
    for emotion in sorted_emotions:
        emotion['emoji'] = emotion_labels_with_emojis.get(emotion['label'], '')  # Get emoji for the label
    
    return sorted_emotions[:3]

def analyze_sentences_with_emotions(text):
    sentences = sent_tokenize(text, language='english')
    sentence_analysis = []
    
    for sentence in sentences:
        sentiment_result = sentiment_pipeline(sentence)[0]
        emotions = analyze_sentence_emotions(sentence)  # Analyze emotions for the sentence
        top_emotion = emotions[0]  # Get the top emotion
        sentence_analysis.append((sentence, sentiment_result['label'], sentiment_result['score'], top_emotion))
    
    return sentence_analysis

def plot_sentiment_bar(score, label):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[2], y=[''], orientation='h',
        marker=dict(color='lightgray', line=dict(color='black', width=1)),  
        base=-1
    ))

    if label == 'positive':
        fig.add_trace(go.Bar(
            x=[score], y=[''], orientation='h',
            marker=dict(color='green', line=dict(color='rgba(0, 0, 0, 0)')),  
            base=0
        ))
    elif label == 'negative':
        fig.add_trace(go.Bar(
            x=[-score], y=[''], orientation='h',
            marker=dict(color='red', line=dict(color='rgba(0, 0, 0, 0)')),  
            base=0
        ))
    else:  # Neutral sentiment
        fig.add_trace(go.Bar(
            x=[0], y=[''], orientation='h',
            marker=dict(color='gray', line=dict(color='rgba(0, 0, 0, 0)')),  
            base=0
        ))

    fig.update_layout(
        xaxis=dict(range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1]),
        showlegend=False,
        height=70,
        margin=dict(l=10, r=10, t=30, b=10),  
        bargap=0,  
        barmode='overlay'  
    )
    
    return fig

def plot_sentence_sentiment_bar(score, label):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[2], y=[''], orientation='h',
        marker=dict(color='lightgray', line=dict(color='black', width=1)),  
        base=-1
    ))

    if label == 'positive':
        fig.add_trace(go.Bar(
            x=[score], y=[''], orientation='h',
            marker=dict(color='green', line=dict(color='rgba(0, 0, 0, 0)')),  
            base=0
        ))
    elif label == 'negative':
        fig.add_trace(go.Bar(
            x=[-score], y=[''], orientation='h',
            marker=dict(color='red', line=dict(color='rgba(0, 0, 0, 0)')),  
            base=0
        ))
    else:  # Neutral sentiment
         fig.add_trace(go.Bar(
            x=[0.01], y=[''], orientation='h',  
            marker=dict(color='gray', line=dict(color='rgba(0, 0, 0, 0)')),  
            base=0
        ))

    fig.update_layout(
        xaxis=dict(range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1]),
        showlegend=False,
        height=50,  
        width=200,  
        margin=dict(l=0, r=0, t=0, b=0),  
        bargap=0,  
        barmode='overlay'  
    )

    return fig

@st.cache_data
def load_amazon_reviews(file_path=AMAZON_PATH, n=200):
    try:
        total_rows = 500000
        
        if total_rows < n:
            st.warning(f"The file contains only {total_rows} rows. Loading all available rows.")
            n = total_rows
        
        # Generate random row indices to skip (the complement of the rows to load)
        random_rows = sorted(random.sample(range(1, total_rows + 1), n))  # Skipping header row
        
        # Load the selected random rows using 'skiprows'
        df = pd.read_csv(file_path, header=None, skiprows=lambda x: x not in random_rows)
        return df
    except Exception as e:
        st.error(f"Error loading random reviews: {e}")
        return None

# Get a random review from the dataset
def get_random_review(df):
    random_idx = random.randint(0, len(df) - 1)
    return df.iloc[random_idx, 2]  # Column 2 contains the full review text

df = load_amazon_reviews()

def main():
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Title
    st.title("Sentiment and Emotional Analysis")

    # Initialize session state
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    st.write("Enter some text below or generate a random Amazon review, then click 'Analyze'!")
    
    if st.button("🔄 Generate random review", key="random"):
        if df is not None:
            st.session_state.user_input = get_random_review(df)

    user_input = st.text_area("", value=st.session_state.user_input, height=150)

    st.markdown('<div class="analyze-button">', unsafe_allow_html=True)
    
    # Button to trigger sentiment analysis
    if st.button("Analyze"):
        highlighted_text = highlight_sentiment_chunks(user_input, sentiment_pipeline)
        
        st.markdown(
            f"""
            <div class="highlighted-text">
                {highlighted_text}
            </div>
            """, unsafe_allow_html=True
        )

        sentiment = sentiment_pipeline(user_input)[0]
        score = sentiment['score']
        label = sentiment['label']

        if label == 'positive':
            label_style = f"<span class='positive-sentiment'>{label.capitalize()}</span>"
        elif label == 'negative':
            label_style = f"<span class='negative-sentiment'>{label.capitalize()}</span>"
        else:
            label_style = f"<span class='neutral-sentiment'>{label.capitalize()}</span>"

        st.markdown(
            f"<div style='text-align:center; font-size:24px; font-weight: 850;'>"
            f"The overall sentiment is {label_style} (Score: {score:.2f}/1.0)</div>",
            unsafe_allow_html=True
        )

        main_sentiment_bar = plot_sentiment_bar(score, label)
        st.plotly_chart(main_sentiment_bar, use_container_width=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        all_emotion = analyze_emotions(user_input)

        # Display the explanatory title
        st.markdown("<div class='emoji-explanation'>Top <span>Emotions</span> Detected from Your Review</div>", unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Display the emotions
        st.markdown("<div class='emotions-container'>", unsafe_allow_html=True)
        for idx, emotion in enumerate(all_emotion[:3]):
            st.markdown(
                f"<div class='emotion-item emotion-item-{idx + 1}'>"
                f"{emotion['label'].capitalize()} <span class='emotion-emoji'>{emotion_labels_with_emojis[emotion['label']]}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Sentence-Level Analysis Section
        sentence_analysis = analyze_sentences_with_emotions(user_input)
        st.write("### Sentence-Level Analysis")

        st.markdown('<div class="sentence-analysis">', unsafe_allow_html=True)
        for sentence, label, score, top_emotion in sentence_analysis:
            st.markdown('<div class="sentence-card">', unsafe_allow_html=True)

            st.markdown(
                f'<div class="sentence-text">'
                f'Sentence: {sentence}</div>', unsafe_allow_html=True
            )

            # Display sentiment label and score 
            st.markdown(
                f'<div class="sentence-info">'
                f'<span class="emotion-label">Emotion</span>: {top_emotion["label"].capitalize()} '
                f'<span class="emotion-emoji">{top_emotion["emoji"]}</span><br>'  
                f'Sentiment: '
                f'<span class="{label.lower()}-sentiment">{label.capitalize()}</span> '
                f'</div>',
                unsafe_allow_html=True
            )
            sentence_bar = plot_sentence_sentiment_bar(score, label)
            unique_key = f"sentence_chart_{idx}_{hash(sentence)}"
            st.plotly_chart(sentence_bar, use_container_width=False, key=unique_key)

            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)



if __name__ == '__main__':
    main()