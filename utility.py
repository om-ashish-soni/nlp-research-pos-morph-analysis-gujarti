def refine(query):
    words = query.split()

    # Join the words back with a single space
    cleaned_text = ' '.join(words)
    
    return cleaned_text