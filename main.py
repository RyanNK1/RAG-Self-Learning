from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI


PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""


DATA_PATH = "data"

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    return splitter.split_documents(documents)

# Load and chunk the documents
documents = load_documents()
chunks = split_text(documents)

#print(f"Loaded {len(documents)} documents and split into {len(chunks)} chunks.")

CHROMA_PATH = "chroma"

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
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

save_to_chroma(chunks)

def query_rag(query_text):
    # Use the same embedding model as before
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load the Chroma DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search for relevant chunks
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Handle no results or low relevance
    if len(results) == 0 or results[0][1] < 0.7:
        return "No relevant results found."

    # Combine the context
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _ in results])

    # Format the prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query_text
    )

    # Set up OpenRouter-compatible LLM
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

    model = ChatOpenAI()

    # Generate the response
    response = model.predict(prompt)
    return response

#query = "Explain how the YOLO method works"
query = "What is the main topic of this document?"
response = query_rag(query)
print("\nResponse:\n", response)

