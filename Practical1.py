import streamlit as st
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

nlp=spacy.load("en_core_web_sm")

st.set_page_config(
    page_title="NLP Pipeline Demo",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>

.stApp{
    background:linear-gradient(to right,#0F172A,#111827);
}

.block-container{
    padding-top:2rem;
}

.stTextArea textarea{
    border-radius:15px;
    border:2px solid #3B82F6;
    font-size:17px;
    background:#1E293B;
}

.stTextArea textarea:focus{
    border:2px solid #60A5FA;
    box-shadow:0 0 15px rgba(96,165,250,.5);
}

div.stButton > button{

width:100%;
height:55px;

background:linear-gradient(90deg,#2563EB,#8B5CF6);

border:none;

border-radius:12px;

color:white;

font-size:18px;

font-weight:bold;

box-shadow:0 8px 20px rgba(37,99,235,.35);

transition:.3s;

}

div.stButton > button:hover{

transform:translateY(-3px);

box-shadow:0 10px 30px rgba(37,99,235,.6);

}

</style>
""",unsafe_allow_html=True)

st.markdown("""
<div style="
    background: white;
    width:55%;
    margin:auto;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.2);
    text-align:center;
    color:#1f2937;
">

<p style="font-size:22px; margin:8px;">
👩 <b>Name:</b> Muskan Pandit
</p>

<p style="font-size:22px; margin:8px;">
📅 <b>Date:</b> 22/07/2026
</p>

</div>
""", unsafe_allow_html=True)


st.markdown("""
<h1 style="
text-align:center;
color:#60A5FA;
font-size:58px;
font-weight:700;
margin-bottom:5px;">
 NLP Pipeline Demo
</h1>
""", unsafe_allow_html=True)


st.markdown("""
<p style="
text-align:center;
font-size:20px;
color:#CBD5E1;">
Enter test and explore NLP steps
</p>
""",unsafe_allow_html=True)

text=st.text_area(
    "Enter your text here:",
    height=170,
    placeholder="Type any paragraph here..."
)

col1,col2,col3=st.columns(3)

with col1:
    st.metric("Characters",len(text))

with col2:
    st.metric("Words",len(text.split()))

with col3:
    if text.strip():
        st.metric("Sentences",len(sent_tokenize(text)))
    else:
        st.metric("Sentences",0)


if st.button("Run NLP Pipeline"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:
        with st.spinner("Analyzing text..."):
            st.success("Analysis Completed Successfully ✅")

        # Sentence Tokenization
        with st.expander("📄 Sentence Tokenization",expanded=True):

            sentences=sent_tokenize(text)

            st.write(sentences)

        # Word Tokenization
        with st.expander("🔤 Word Tokenization"):

            words=word_tokenize(text)

            st.write(words)

        # Stopword Removal
        with st.expander("🚫 Stopword Removal"):

            stop_words=set(stopwords.words("english"))

            filtered_words=[word for word in words if word.lower() not in stop_words]

            st.success(filtered_words)

        # POS Tagging
        with st.expander("🏷 Part-of-Speech Tagging"):

            tagged_words=nltk.pos_tag(words)

            st.code(tagged_words)

        # Lemmatization
        with st.expander("📖 Lemmatization"):

            lemmatizer=WordNetLemmatizer()

            lemmas=[lemmatizer.lemmatize(word) for word in filtered_words]

            st.write(lemmas)

        # Named Entity Recognition
        with st.expander("🌍 Named Entity Recognition"):

            doc=nlp(text)

            if doc.ents:

                for ent in doc.ents:

                    st.info(f"**{ent.text}** → {ent.label_}")

            else:

                st.warning("No named entities found.")

        # Token Details
        with st.expander("📊 Token Details",expanded=True):

            token_data=[]

            for token in doc:

                token_data.append({

                    "Token":token.text,

                    "Lemma":token.lemma_,

                    "POS":token.pos_,

                    "Dependency":token.dep_

                })

            df=pd.DataFrame(token_data)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )    

    

    

        