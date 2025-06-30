import os
import shutil
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Constants
DATA_PATH = "data"
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""

# Step 1: Load PDF documents
def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

# Step 2: Split documents into chunks
def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    return splitter.split_documents(documents)

# Step 3: Save chunks to Chroma vector store
def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        chunks,
        embedding_function,
        persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

# Step 4: Query the RAG system
def query_rag(query_text):
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0 or results[0][1] < 0.1:
       return "No relevant results found."

    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _ in results])[:1000]  # limit to 1000 characters

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    messages = prompt_template.format_messages(
        context=context_text,#SHorten context text
        question=query_text
    )

    # Debug: Check if API key is loaded
    print("DEBUG: OPENROUTER_API_KEY =", os.getenv("OPENROUTER_API_KEY"))

    model = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        #model="mistralai/mistral-7b-instruct"
    )

    #response = model.invoke(messages)
    response=model.chat.completions.create(
        model="mistralai/mistral-7b-instruct",#gpt-4.1-nano gpt-4.1-mini gpt-4.1
        messages=messages,
        max_tokens = 1000
    )
    return response.choices[0].message.content

# Main execution
if __name__ == "__main__":
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

    query = "Academic Strategies"
    response = query_rag(query)
    print("\nResponse:\n", response)


