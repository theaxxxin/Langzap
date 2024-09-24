import streamlit as st
from langzap import Langzap

# Initialize Langzap
zap = Langzap()

st.title("Langzap Streamlit App")
st.write("This app showcases the functionalities of the Langzap library.")

# Function to display the result
def display_result(result):
    st.write("### Result")
    st.write(result)

# Ask Function
st.header("Ask Function")
prompt = st.text_input("Enter your prompt:")
if st.button("Ask"):
    response = zap.ask(prompt)
    display_result(response)

# Summarize Function
st.header("Summarize Function")
text_to_summarize = st.text_area("Enter text to summarize:")
max_words = st.number_input("Max words for summary:", min_value=1, value=50)
if st.button("Summarize"):
    summary = zap.summarize(text_to_summarize, max_words)
    display_result(summary)

# Sentiment Analysis Function
st.header("Sentiment Analysis Function")
text_for_sentiment = st.text_area("Enter text for sentiment analysis:")
if st.button("Analyze Sentiment"):
    sentiment = zap.sentiment(text_for_sentiment)
    display_result(sentiment)

# Entity Extraction Function
st.header("Entity Extraction Function")
text_for_extraction = st.text_area("Enter text for entity extraction:")
entity_types = st.text_input("Enter entity types to extract (comma-separated):")
if st.button("Extract Entities"):
    entities = zap.extract(text_for_extraction, entity_types.split(","))
    display_result(entities)

# Research Function
st.header("Research Function")
query = st.text_input("Enter your research query:")
num_results = st.number_input("Number of search results:", min_value=1, value=3)
if st.button("Research"):
    research_result = zap.research(query, num_results)
    display_result(research_result)

# Translation Function
st.header("Translation Function")
text_to_translate = st.text_area("Enter text to translate:")
target_language = st.text_input("Enter target language:")
if st.button("Translate"):
    translation = zap.translate(text_to_translate, target_language)
    display_result(translation)
