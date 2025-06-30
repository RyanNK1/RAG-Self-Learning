# RAG-Self-Learning Notes

SECTION 1: Imports and Environment Setup

🔹 Line 1: import os
✅ What it does:
Imports Python’s built-in os module, which provides a way to interact with the operating system.
🧠 Why it's important:
You’ll use os to:
 1. Check if directories exist (os.path.exists)
 2. Access environment variables (os.getenv)
 3. Work with file paths
📌 Example usage in main.py:

#  if os.path.exists(CHROMA_PATH):
#     shutil.rmtree(CHROMA_PATH)


This checks if the Chroma directory exists and deletes it if it does.

🔹 Line 2: import shutil
✅ What it does:
Imports the shutil module, which stands for shell utilities.
🧠 Why it's important:
It allows you to perform high-level file operations, such as:
1. Copying files and directories
2. Deleting entire directories (shutil.rmtree)
3. Moving files
📌 Example usage in main.py:

# shutil.rmtree(CHROMA_PATH)

This deletes the entire Chroma directory and its contents to ensure a clean start.

🔹 Line 3: from dotenv import load_dotenv
✅ What it does:
Imports the load_dotenv function from the python-dotenv package.
🧠 Why it's important:
1. It loads environment variables from a .env file into your Python environment.
This is a best practice for managing API keys, secrets, and configuration without hardcoding them into your code.
📌 Example usage in main.py:

# load_dotenv()

This line (called later) ensures that variables like OPENROUTER_API_KEY are available via os.getenv.

🧠 Concept: What is a .env file?

A .env file is a plain text file that contains key-value pairs like:

OPENROUTER_API_KEY= Check .env File

This keeps sensitive information out of your codebase and allows different environments (dev, test, prod) to use different settings.

🧠 LangChain and Related Libraries

📌 Example usage in main.py:

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_openai import ChatOpenAI
# from openai import OpenAI
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.prompts import ChatPromptTemplate

Concepts:

HuggingFaceEmbeddings: Uses Hugging Face models to convert text into vector embeddings.
Chroma: A vector store that stores and retrieves embeddings efficiently.
ChatOpenAI: LangChain wrapper for OpenAI chat models (not used directly here).
OpenAI: Direct SDK access to OpenRouter-compatible models.
PyPDFDirectoryLoader: Loads all PDFs from a directory and extracts text.
RecursiveCharacterTextSplitter: Splits long texts into manageable chunks with overlap.
ChatPromptTemplate: Helps format prompts for chat models using templates.

🧩 SECTION 2: Constants and Prompt Template

Usage in Code: 

# DATA_PATH = "data"
# CHROMA_PATH = "chroma"
# PROMPT_TEMPLATE = """
# Answer the question based only on the following      context:
# {context}
#  - -
# Answer the question based on the above context: {question}
# """

🔹 Line 1: DATA_PATH = "data"
✅ What it does:
Defines a string constant that points to the folder where your PDF documents are stored.
🧠 Why it's important:
This path is used by the PyPDFDirectoryLoader to locate and load documents.
Keeping it as a constant makes it easy to change the folder location later without modifying multiple parts of the code.
🔹 Line 2: CHROMA_PATH = "chroma"
✅ What it does:
Defines the path where the Chroma vector store will be saved.
🧠 Why it's important:
Chroma uses this directory to persist embeddings so they can be reused across sessions.
If you restart your app, you don’t need to reprocess documents — just reload from this path.
🔹 Line 3–7: PROMPT_TEMPLATE = """..."""
✅ What it does:
Defines a multi-line string template for the prompt that will be sent to the language model.
🧠 Why it's important:
This template is used to structure the input to the LLM in a way that guides its behavior.
It includes two placeholders:
{context}: Injects the retrieved document chunks.
{question}: Injects the user’s query.
📌 Example of how it works:
If the context is:

Students should manage their time effectively by using planners and setting goals.
And the question is:

What are good academic strategies?
The final prompt becomes:

Answer the question based only on the following context:
Students should manage their time effectively by using planners and setting goals.
 - -
Answer the question based on the above context: What are good academic strategies?
This ensures the model only uses the provided context to answer the question, which is the core idea of retrieval-augmented generation.

🧩 SECTION 3: Load PDF Documents

# from langchain_community.document_loaders import PyPDFDirectoryLoader

# def load_documents():
#    loader = PyPDFDirectoryLoader(DATA_PATH)
#    return loader.load()


🔹 Line 1: from langchain_community.document_loaders import PyPDFDirectoryLoader

✅ What it does:
Imports the PyPDFDirectoryLoader class from LangChain’s community-maintained document loaders.
🧠 Concept:
Document loaders are utilities in LangChain that help you ingest raw data (PDFs, text files, web pages, etc.) and convert them into structured Document objects.
PyPDFDirectoryLoader is specifically designed to:
Traverse a directory
Find all .pdf files
Extract text from each page
Return a list of Document objects
Each Document object contains:

page_content: the extracted text
metadata: file name, page number, etc.

🔹 Line 2: def load_documents():

✅ What it does:
Defines a function named load_documents that encapsulates the logic for loading PDFs.
🧠 Why use a function?
Encapsulation improves readability and reusability.
You can call load_documents() anywhere in your code to get the documents without repeating logic.

🔹 Line 3: loader = PyPDFDirectoryLoader(DATA_PATH)
✅ What it does:
Creates an instance(Of the class) of PyPDFDirectoryLoader, pointing it to the directory defined by DATA_PATH.
🧠 Concept:
DATA_PATH is "data", so this line tells the loader to look inside the data/ folder for PDF files.
The loader will automatically:
Open each PDF
Read its contents
Prepare it for downstream processing

🔹 Line 4: return loader.load()
✅ What it does:
Calls the .load() method on the loader, which:
Reads all PDFs
Extracts text
Returns a list of Document objects
🧠 Output:
A list like:

# [
#   Document(page_content="Text from page 1", metadata={"source": "file1.pdf", "page": 1}),
#   Document(page_content="Text from page 2", metadata={"source": "file1.pdf", "page": 2}),
#  ...
# ]


This output is the raw material for your RAG system — it’s what you’ll split, embed, and retrieve later.

🧩 SECTION 4: Split Documents into Chunks

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# def split_text(documents):
#    splitter = RecursiveCharacterTextSplitter(
#        chunk_size=300,
#        chunk_overlap=100,
#        length_function=len,
#        add_start_index=True
#    )
#    return splitter.split_documents(documents)


🔹 Line 1: from langchain.text_splitter import RecursiveCharacterTextSplitter
✅ What it does:
Imports the RecursiveCharacterTextSplitter class from LangChain.
🧠 Concept:
Text splitting is necessary because LLMs have token limits (e.g., 4,096 tokens for GPT-3.5).
This class intelligently splits long documents into smaller chunks that:
Fit within token limits
Preserve semantic meaning
Allow for overlapping context between chunks
🔹 Line 2: def split_text(documents):
✅ What it does:
Defines a function named split_text that takes a list of Document objects as input.
🧠 Why use a function?
Encapsulates the logic for splitting documents.
Makes the code reusable and modular.
🔹 Line 3–7: splitter = RecursiveCharacterTextSplitter(...)
✅ What it does:
Creates an instance of the RecursiveCharacterTextSplitter with specific configuration.
🧠 Parameters Explained:
Parameter ->	Purpose
chunk_size=300	Each chunk will contain up to 300 characters
chunk_overlap=100	Each chunk will overlap with the previous one by 100 characters
length_function=len	Uses Python’s len() to measure chunk size in characters
add_start_index=True	Adds metadata indicating where each chunk starts in the original document
🧠 Why overlap?
Overlap ensures that important context isn’t lost between chunks.
For example, if a sentence spans two chunks, overlap helps preserve continuity.
🔹 Line 8: return splitter.split_documents(documents)
✅ What it does:
Applies the splitter to the list of documents.
Returns a new list of smaller Document chunks.
🧠 Output:
Each chunk is a Document object with:
1. page_content: the chunked text
2. metadata: including start index, source file, etc.

This prepares the data for embedding and storage in the next step.

✅ Summary of This Section
Line	Purpose
RecursiveCharacterTextSplitter	Splits long documents into manageable chunks
chunk_size	Controls the size of each chunk
chunk_overlap	Ensures context continuity between chunks
split_documents()	Returns a list of chunked Document objects