import streamlit as st
import PyPDF2
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from tempfile import NamedTemporaryFile

# Ingest PDF
def ingest(local_path):
    if local_path:
        pdf_reader = PyPDF2.PdfReader(local_path)
        extracted_text = []
        for page in pdf_reader.pages:
            extracted_text.append(page.extract_text())
        doc = [Document(page_content=document) for document in extracted_text]
        return doc
    else:
        return "Upload a PDF file"

# Create vector embeddings
def vector_embeddings(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="local-rag"
    )
    return vector_db

# Define retriever function
def retriever(vector_db):
    local_model = "llama3.2"
    llm = ChatOllama(model=local_model)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. 
        Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. 
        By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the 
        distance-based similarity search. Provide these alternative questions separated by newlines. Original questions: {question}"""
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

    template = """Answer the questions based ONLY on the following context:
    {context}
    Question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# Streamlit UI
def main():
    st.title("Medical PDF Analysis")
    st.write("Upload a medical PDF report to generate a diagnosis and suggestions.")

    uploaded_file = st.file_uploader("Upload Medical PDF", type="pdf")

    if uploaded_file is not None:
        # Display uploaded file name

        # Ingest the PDF
        with st.spinner("Reading and processing the PDF..."):
            data = ingest(uploaded_file)
            
            if isinstance(data, str):
                st.error(data)

        # Create vector embeddings
        with st.spinner("Generating vector embeddings..."):
            vector_db = vector_embeddings(data)

        # Retrieve and analyze the data
        with st.spinner("Analyzing the report and generating suggestions..."):
            chain = retriever(vector_db)
            result = chain.invoke("""Read the medical report and answer the following questions with proper headings:
                                    1. What condition or illness does this indicate?
                                    2. What could have caused this condition?
                                    3. Are there any complications or risks I should be aware of?
                                    4. What treatments or medications are recommended?
                                    5. What can I do to improve my health based on these results?
                                    6. Do I need to see a specialist? If so, which type?""")
            st.subheader("Diagnosis and Suggestions:")
            st.write(result)

if __name__ == "__main__":
    main()
