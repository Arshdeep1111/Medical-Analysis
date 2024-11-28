# Medical PDF Analysis Application

This is a Python-based Streamlit application designed for analyzing medical PDFs. It processes medical reports, extracts information, and provides diagnostic suggestions using advanced AI models.

---

## Features

- **Upload and Process Medical PDFs**  
  Ingests a PDF and extracts its textual content for analysis.

- **Vector Embeddings Creation**  
  Utilizes `Chroma` and `OllamaEmbeddings` to generate vector representations of the document.

- **Custom Query Retrievers**  
  Employs a multi-query retriever mechanism to ensure comprehensive and relevant document retrieval.

- **AI-Powered Analysis**  
  Analyzes the extracted data using a language model (`ChatOllama`) and generates:
  - Diagnosis
  - Causes
  - Risks and complications
  - Treatment recommendations
  - Health improvement suggestions

- **User-Friendly Interface**  
  Built with Streamlit for an intuitive and interactive user experience.

