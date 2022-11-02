import os

import streamlit as st

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    pipeline
)

st.title("Question Answering with roBERTa")
st.markdown("")
st.markdown("")


form = st.form(key="my_form")

# creating the q/a pipeline
nlp = pipeline(
    "question-answering", 
    model="deepset/roberta-base-squad2", 
    tokenizer="deepset/roberta-base-squad2"
)

text = form.text_area("Provide a text to answer questions from", height=200)

submit_button = form.form_submit_button(label="Study This")

st.markdown("---")
ques=st.text_input("Ask Me Anything From The Information You Have Given")

#forming a question directory 
ques_dict = {
    "question": ques,
    "context": text
}

if st.button("Ask"):
    results = nlp(ques_dict)
    st.markdown("---")
    st.subheader("Here Is Your Answer")
    st.success(results["answer"])
    st.balloons()