import streamlit as st
import spacy
import spacy_streamlit as spt
nlp = spacy.load('en_core_web_sm')

def main():
    st.title('NLP (Natural Language Processing) APP')
    menu = ['Home', 'Tokenization', 'POS Tagging', 'NER', 'Lemmatization']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Choose a task from the menu')
    
    elif choice == 'Tokenization':
        st.subheader('Word Tokenization')
        raw_text = st.text_area('Enter Text', 'Enter your text here...')
        docs = nlp(raw_text)
        if st.button('Tokenize'):
            spt.visualize_tokens(docs)

    elif choice == 'POS Tagging':
        st.subheader('Part-of-Speech Tagging')
        raw_text = st.text_area('Enter Text', 'Enter your text here...')
        docs = nlp(raw_text)
        tokens_data = [{'text': token.text, 'pos': token.pos_, 'dep': token.dep_} for token in docs]
        st.table(tokens_data)

    elif choice == 'NER':
        st.subheader('Named Entity Recognition')
        raw_text = st.text_area('Enter Text', 'Enter your text here...')
        docs = nlp(raw_text)
        if st.button('Recognize Entities'):
            spt.visualize_ner(docs, labels=nlp.get_pipe('ner').labels)

    elif choice == 'Lemmatization':
        st.subheader('Lemmatization')
        raw_text = st.text_area('Enter Text', 'Enter your text here...')
        docs = nlp(raw_text)
        lemmatized_tokens = [token.lemma_ for token in docs]
        st.write('Lemmatized Tokens:', lemmatized_tokens)

if __name__ == '__main__':
    main()
