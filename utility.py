import streamlit as st
from samples import sentences
import random

def refine(query):
    words = query.split()

    # Join the words back with a single space
    cleaned_text = ' '.join(words)
    
    return cleaned_text

def take_input_query():
    selected_option = st.selectbox("Select a sample sentence:", sentences)
    query = st.text_input("Or Enter your sentence in Gujarati here...", "")
    # Use the selected_option if query is empty
    if not query:
        query = selected_option
    query=refine(query)

    return query

