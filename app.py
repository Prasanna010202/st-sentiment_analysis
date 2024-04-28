import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import altair as alt


# Load NLTK resources
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')

# Load RoBERTa model and tokenizer
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL) 
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Function to perform sentiment analysis using VADER model
def perform_vader_sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

# Function to perform sentiment analysis using RoBERTa model
def perform_roberta_sentiment_analysis(text):
    encoded_text = tokenizer(text, return_tensors="pt")
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {
        "roberta_neg": scores[0],
        "roberta_neu": scores[1],
        "roberta_pos": scores[2]
    }

# Function to preprocess and analyze sentiments
def preprocess_and_analyze_sentiments(df):
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = row["Text"]
            myid = row["Id"]
            vader_result = perform_vader_sentiment_analysis(text)
            vader_result_rename = {f"vader_{key}": value for key, value in vader_result.items()}
            roberta_result = perform_roberta_sentiment_analysis(text)
            both = {**vader_result_rename, **roberta_result}
            res[myid] = both
        except RuntimeError:
            print(f"Broke for id {myid}")
    return res

def calculate_metrics(actual, predicted):
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, average='weighted')
    recall = recall_score(actual, predicted, average='weighted')
    f1 = f1_score(actual, predicted, average='weighted')
    return accuracy, precision, recall, f1

# Load dataset


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page:", ["Home", "Sentiment Analysis"])
    df = pd.read_csv("archive/Reviews.csv")
    df = df.head(1000)  # Limiting the dataset size for demonstration

    if page == "Home":
        st.title("Review Sentiments Unleashed: Explore the Emotions Behind the Words")
        res = preprocess_and_analyze_sentiments(df)
        results_df = pd.DataFrame(res).T
        results_df = results_df.reset_index().rename(columns={"index":"Id"})
        results_df = results_df.merge(df, how="left")

        # Visualize sentiments
        st.title("Comparison of Sentiment Analysis Models")
        st.write("Visualizing Sentiments")
        st.write(results_df.head(100))  # Display a sample of the dataframe

        # Scatter plots for comparing VADER and RoBERTa sentiments
        st.subheader("Comparison of Positive Sentiments")
        st.scatter_chart(results_df[['roberta_pos', 'vader_pos']], color=("#080e42", "#f2dbaf"))
        st.subheader("Comparison of Negative Sentiments")
        st.scatter_chart(results_df[['roberta_neg', 'vader_neg']], color=("#610030","#d2b48c"))

        scatter = alt.Chart(results_df).mark_circle().encode(
            x='vader_pos',
            y='roberta_pos',
            color=alt.Color('vader_neg', scale=alt.Scale(scheme='yellowgreenblue')),
            size='vader_neg',
            tooltip=['Text']).properties(width=600, height=400).interactive()

        st.altair_chart(scatter, use_container_width=True)
        
    elif page == "Sentiment Analysis":
        st.title("Analyzing Text Emotions")
        
        # Input Text
        text_input = st.text_area("Enter text for sentiment analysis:", "")
        
        if st.button("Analyze Sentiment"):
            # Perform sentiment analysis using both models
            vader_result = perform_vader_sentiment_analysis(text_input)
            roberta_result = perform_roberta_sentiment_analysis(text_input)

            # Displaying Results
            st.subheader("VADER Model Results:")
            st.write(vader_result)

            st.subheader("RoBERTa Model Results:")
            st.write(roberta_result)

            # Visualize Results
            st.subheader("Visualization of Sentiment Scores:")
            st.write("VADER vs. RoBERTa")
            df = pd.DataFrame({
                "Sentiment": ["Negative", "Neutral", "Positive"],
                "VADER Score": [vader_result["neg"], vader_result["neu"], vader_result["pos"]],
                "RoBERTa Score": [roberta_result["roberta_neg"], roberta_result["roberta_neu"], roberta_result["roberta_pos"]]
            })
            st.bar_chart(df.set_index("Sentiment"))

            # Confidence Levels
            st.subheader("Confidence Levels:")
            st.write("VADER Compound Score:", vader_result["compound"])
            st.write("RoBERTa Positive Score:", roberta_result["roberta_pos"])
            st.write("RoBERTa Negative Score:", roberta_result["roberta_neg"])

            
if __name__ == "__main__":
    main()
