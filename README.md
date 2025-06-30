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

🧩 SECTION 5: Save Chunks to Chroma Vector Store

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# import os, shutil

# def save_to_chroma(chunks):
#    if os.path.exists(CHROMA_PATH):
#        shutil.rmtree(CHROMA_PATH)


🔹 Line 1–2: Importing Required Modules
HuggingFaceEmbeddings: A LangChain wrapper that uses Hugging Face models to convert text into vector embeddings.
Chroma: A vector store that allows you to store and search those embeddings efficiently.
🔹 Line 3: import os, shutil
These were imported earlier, but are used again here to manage the file system:
os.path.exists(...): Checks if the Chroma directory already exists.
shutil.rmtree(...): Deletes the directory and all its contents.
🔹 Line 5–6: Check and Clear Existing Vector Store

# if os.path.exists(CHROMA_PATH):
#    shutil.rmtree(CHROMA_PATH)

✅ What it does:
If a Chroma vector store already exists at the specified path, it deletes it.
🧠 Why?
Ensures you're starting with a clean slate.
Prevents mixing old and new data, which could lead to incorrect retrieval results.

🔹 Line 7–9: Create Embedding Function

# embedding_function = HuggingFaceEmbeddings(
#    model_name="sentence-transformers/all-MiniLM-L6-v2"
# )


✅ What it does:
Initializes an embedding model from Hugging Face.
🧠 Concept: What are Embeddings?
Embeddings are numerical representations of text.
They allow you to compare the semantic similarity between pieces of text using vector math.
The model "all-MiniLM-L6-v2" is a lightweight, fast, and accurate model for generating sentence embeddings.
🔹 Line 10–13: Create and Persist Chroma Vector Store

# db = Chroma.from_documents(
#     chunks,
#     embedding_function,
#     persist_directory=CHROMA_PATH
# )

✅ What it does:
Converts each chunk into a vector using the embedding model.
Stores those vectors in a Chroma vector database.
Saves the database to disk at the path defined by CHROMA_PATH.
🧠 Concept: What is a Vector Store?
A vector store is a specialized database that:
Stores high-dimensional vectors
Supports similarity search (e.g., find the most similar chunks to a query)
Chroma is a local, lightweight vector store that integrates well with LangChain.
🔹 Line 14: Print Confirmation

# print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

Confirms how many chunks were embedded and stored.
✅ Summary of This Section
Line	Purpose
HuggingFaceEmbeddings	Converts text into vector form using a Hugging Face model
Chroma.from_documents	Stores those vectors in a searchable database
shutil.rmtree	Clears old vector data to avoid conflicts
persist_directory	Saves the vector store to disk for reuse

🧩 SECTION 6: Query the RAG System

🔹 Line 1–2: Import Required Modules

# from langchain.prompts import ChatPromptTemplate
# from openai import OpenAI

ChatPromptTemplate: A LangChain utility for formatting prompts in a structured way for chat models.
OpenAI: The SDK used to interact with OpenRouter-compatible models (like Mistral or GPT-4.1).
🔹 Function Definition

# def query_rag(query_text):

Defines a function that takes a user query (string) and returns a generated answer.
This function performs retrieval + generation, the core of RAG.

🔹 Re-initialize Embedding Function

# embedding_function = HuggingFaceEmbeddings(
#    model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

Creates a new instance of the embedding model.
This is used to convert the query into a vector for similarity search.
🔹 Load Chroma Vector Store

# db = Chroma(
#    persist_directory=CHROMA_PATH,
#    embedding_function=embedding_function
# )

Loads the previously saved Chroma vector store from disk.
Associates it with the same embedding function to ensure compatibility.
🔹 Perform Similarity Search

# results = db.similarity_search_with_relevance_scores(query_text, k=3)

Searches for the top 3 chunks most similar to the query.
Returns a list of tuples: (Document, relevance_score).

🧠 Concept: Similarity Search
Compares the query vector to stored document vectors.
Uses cosine similarity or other metrics to rank relevance.
🔹 Filter Low-Relevance Results

# if len(results) == 0 or results[0][1] < 0.1:
#     return "No relevant results found."

If no results are found or the top result has a low relevance score (< 0.1), return a fallback message.
🔹 Construct Context from Results

# context_text = "\n\n - -\n\n".join([doc.page_content for doc, _ in results])[:1000]

Joins the top 3 chunks into a single string.
Limits the context to 1000 characters to stay within token limits.
🔹 Format Prompt Using Template

# prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
# messages = prompt_template.format_messages(
#     context=context_text,
#     question=query_text
# )

Uses the earlier-defined PROMPT_TEMPLATE to format the prompt.
Injects the retrieved context and user question into the template.
Produces a list of messages suitable for chat-based models.
🔹 Debug: Print API Key

# print("DEBUG: OPENROUTER_API_KEY =", os.getenv("OPENROUTER_API_KEY"))

Prints the API key (if loaded) to verify that environment variables are working.
🔹 Initialize OpenRouter Model

# model = OpenAI(
#    base_url="https://openrouter.ai/api/v1",
#    api_key=os.getenv("OPENROUTER_API_KEY"),
# )


Creates an instance of the OpenAI client configured to use OpenRouter.
Uses the API key from the .env file.
🔹 Send Prompt and Get Response

# response = model.chat.completions.create(
#     model="mistralai/mistral-7b-instruct",
#     messages=messages,
#     max_tokens=1000
# )


Sends the formatted prompt to the specified model.
Requests a response with a maximum of 1000 tokens.
🔹 Return the Final Answer

# return response.choices[0].message.content

Extracts and returns the generated answer from the response object.

✅ Summary of This Section
Line	Purpose
embedding_function	Converts query into a vector
Chroma(...)	Loads the vector store
similarity_search_with_relevance_scores	Finds relevant chunks
context_text	Builds context from retrieved chunks
ChatPromptTemplate	Formats the prompt for the LLM
OpenAI(...)	Initializes the model client
model.chat.completions.create(...)	Sends prompt and gets response
return ...	Returns the generated answer

🤖 Deep Dive: AI Integration in The RAG System
🔌 1. Connecting to OpenRouter via the OpenAI SDK

# model = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPENROUTER_API_KEY"),
# )


✅ What’s happening here:
You’re using the OpenAI SDK, but pointing it to OpenRouter’s API endpoint.
base_url: Tells the SDK to send requests to OpenRouter instead of OpenAI.
api_key: Authenticates your requests using a key stored in your .env file.
🧠 Why this works:
OpenRouter is API-compatible with OpenAI’s chat models.
This means you can use the same SDK and method calls (chat.completions.create) to interact with models like:
mistralai/mistral-7b-instruct
gpt-4.1-mini
meta-llama/llama-3-8b-instruct
🧾 2. Sending a Chat Prompt to the Model

# response = model.chat.completions.create(
#    model="mistralai/mistral-7b-instruct",
#    messages=messages,
#    max_tokens=1000
# )

✅ What’s happening here:
You’re sending a chat-style prompt to the model.
model: Specifies which LLM to use.
messages: A list of structured messages (system/user roles).
max_tokens: Limits the length of the response to avoid excessive output.
🧠 Why chat format?
Chat models expect input as a conversation:
system: Sets behavior or instructions.
user: Asks a question.
assistant: Responds.
Your ChatPromptTemplate generates this format automatically.

🧠 Example of messages Structure

This structure helps the model understand:

What context it should use
What question it should answer
What behavior it should follow (e.g., not hallucinate beyond the context)
📤 3. Extracting the Model’s Response

# return response.choices[0].message.content

✅ What’s happening here:
The model returns a response object with one or more choices.
Each choice contains a message with the generated content.
You extract the first choice and return its content.
🔐 Security Note
Using os.getenv("OPENROUTER_API_KEY") ensures:

Your API key is not hardcoded in the script.
You can easily switch keys or environments by changing the .env file.
✅ Summary of AI Integration
Component	Role
OpenAI client	Connects to OpenRouter
base_url	Redirects SDK to OpenRouter
api_key	Authenticates your requests
messages	Structured prompt for the LLM
chat.completions.create()	Sends prompt and gets response
response.choices[0].message.content	Extracts the final answer

🧠 What Does SDK Mean?
SDK stands for Software Development Kit.

🔹 Definition:
An SDK is a collection of tools, libraries, documentation, and code samples that developers use to build software applications for a specific platform or service.

🔍 In Your Case: OpenAI SDK
When you use this line:

# from openai import OpenAI


You're using the OpenAI SDK, which is a Python package that provides:

Pre-built methods to interact with OpenAI or OpenRouter APIs
Easy access to models like GPT-4, Mistral, Claude, etc.
Built-in handling for authentication, formatting, and response parsing
🧠 Why Use an SDK?
Without an SDK, you'd have to:

Manually format HTTP requests
Handle authentication headers
Parse JSON responses
Deal with errors and retries
With an SDK, you can simply write:

response = model.chat.completions.create(...)


And the SDK takes care of all the underlying complexity.

✅ Summary
Term	Meaning
SDK	Software Development Kit — a set of tools to help developers interact with a platform
OpenAI SDK	A Python toolkit to easily use OpenAI or OpenRouter models
Benefit	Simplifies coding, reduces errors, and speeds up development

🧩 SECTION 7: Main Execution Block

🔹 Line 1: if __name__ == "__main__":
✅ What it does:
This is a Python idiom that ensures the code inside this block only runs when the script is executed directly (not imported as a module).
🧠 Why it matters:
If you later import this script into another file, this block won’t run automatically.
It’s a way to define the entry point of your program.
🔹 Line 2: documents = load_documents()
✅ What it does:
Calls the load_documents() function to read and extract text from all PDFs in the data/ folder.
🧠 Output:
A list of Document objects, each containing text and metadata.
🔹 Line 3: chunks = split_text(documents)
✅ What it does:
Passes the loaded documents to split_text() to break them into smaller, overlapping chunks.
🧠 Output:
A list of chunked Document objects, ready for embedding.
🔹 Line 4: save_to_chroma(chunks)
✅ What it does:
Embeds the chunks using Hugging Face embeddings.
Stores them in a Chroma vector database on disk.
🧠 Result:
You now have a searchable knowledge base of your PDF content.
🔹 Line 5: query = "Academic Strategies"
✅ What it does:
Defines a sample query string that you want to ask the system.
🧠 You can replace this with:
User input from a UI
A command-line argument
A dynamic query from another system
🔹 Line 6: response = query_rag(query)
✅ What it does:
Sends the query to your RAG system:
Retrieves relevant chunks from Chroma
Formats a prompt with those chunks
Sends it to the language model via OpenRouter
Returns the generated answer
🔹 Line 7: print("\nResponse:\n", response)
✅ What it does:
Displays the final answer from the model in the console.
✅ Summary of This Section
Line	Purpose
if __name__ == "__main__"	Ensures the code runs only when executed directly
load_documents()	Loads and extracts text from PDFs
split_text()	Splits documents into manageable chunks
save_to_chroma()	Embeds and stores chunks in a vector database
query_rag()	Retrieves relevant chunks and generates an answer
print()	Displays the result