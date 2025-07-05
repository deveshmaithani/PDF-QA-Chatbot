# 📄 RAG PDF QA Chatbot

An end-to-end **Retrieval-Augmented Generation (RAG)** based chatbot that answers questions based on the content of a PDF using open-source models. Built with **LangChain**, **Hugging Face Transformers**, **FAISS**, and **Streamlit** — no OpenAI API required!

---

## 🚀 Features

- 🔍 Reads and indexes a PDF into semantic chunks
- 🧠 Uses local Hugging Face models (`flan-t5-base`) for generating answers
- 📚 Retrieves relevant context from PDF using FAISS vector store
- 🧾 Custom prompt for precise, context-aware answers
- 💡 Interactive web interface with Streamlit
- ✅ Completely offline & free — no API key required

---

## 🧱 Project Structure

rag_pdf_qa_bot/
├── app.py # Streamlit app UI
├── ingest.py # PDF reading, chunking, and FAISS indexing
├── qa_chain.py # RetrievalQA chain setup with prompt
├── requirements.txt # All dependencies
├── sample.pdf # Your input PDF (you can replace this)
└── faiss_index/ # Saved vector store (generated after ingest)

✍️ Author
Devesh Maithani
Full Stack Developer | Aspiring Data Scientist