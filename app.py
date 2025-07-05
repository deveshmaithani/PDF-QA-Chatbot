import streamlit as st
from qa_chain import load_qa_chain

st.title("📄 PDF QA Chatbot")
st.write("Ask questions based on the uploaded PDF")

qa_chain = load_qa_chain()

query = st.text_input("Enter your question:")
if query:
    with st.spinner("Searching for answers..."):
        response = qa_chain({"query": query})
        st.success(response["result"])