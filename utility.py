import streamlit as st
import pandas as pd
from metadata import feature_meanings

def display_feature_value_meanings(output_feature_values):
    df = pd.DataFrame(columns=["Feature Value", "Meaning"])
    row_index=0
    for value in output_feature_values:
        if value in feature_meanings:
            df.loc[row_index]=[value,feature_meanings[value]]
        row_index+=1
        pass
    
    # Add horizontal rule
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### Feature Value Definitions:")

    # Convert DataFrame to HTML table
    df_html = df.to_html(index=False, escape=False)

    # Display the HTML table
    st.write(df_html, unsafe_allow_html=True)
