import streamlit as st

def main():
    
    st.set_page_config(
        page_title="NLP Gujarati POS Tagging & Morph Analyzer",
        page_icon="✨",
        
    )
    
    
if __name__ == "__main__":
    main()

pages={
    "Home":"home",
    "GUJ POS MORPH":"pos_morph_model", 
    "GUJ ONLY MORPH":"morph_by_pos_support", 
    "GUJ ONLY POS":"only_pos_model"
}

# Render the navigation links
page = st.sidebar.radio("Navigation", list(pages.keys()))

def styled_text(text, font_weight="bold", font_size="large", color="#333"):
    return f'<span style="font-weight: {font_weight}; font-size: {font_size}; color: {color};">{text}</span>'

# Dynamically render the selected page
if page == "Home":
    st.title("NLP Gujarati POS Tagging & Morph Analyzer")
    st.markdown("---")
    st.header("About")
    st.markdown(
        styled_text("This application allows you to perform part-of-speech tagging and morphological analysis", font_size="large") + " "
        + styled_text("of Gujarati language text using different models.", font_size="large"),
        unsafe_allow_html=True 
    )
    st.markdown(
        styled_text("You can select the desired model from the navigation sidebar to get started.", color="#007bff") + " "
        + styled_text("Each model serves a specific purpose based on your requirements.", color="#007bff"),
        unsafe_allow_html=True 
    )
    st.subheader("Models Available:")
    st.markdown("- " + styled_text("GUJ POS MORPH:", color="#28a745") + " Model for predicting POS and morphological features.",unsafe_allow_html=True )
    st.markdown("- " + styled_text("GUJ ONLY MORPH:", color="#dc3545") + " Model for predicting morphological features with POS support.",unsafe_allow_html=True )
    st.markdown("- " + styled_text("GUJ ONLY POS:", color="#007bff") + " Model for predicting only POS of Gujarati words.",unsafe_allow_html=True )

else:
    # Import and render the selected file
    file_path = pages[page] + ".py"
    with open(file_path, "r", encoding="utf-8") as f:
        exec(f.read())
