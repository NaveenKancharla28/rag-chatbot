import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(" API key not found! Make sure you have a .env file.")

print(" OpenAI API key loaded successfully!")

import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI  # Correct import for chat models


from langchain.chains import RetrievalQA


# we are loading the api key
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API Key not there in the code")

print("OPEN_API_KEY is loaded successfully")

#step 2 LOAD TEXT THAT IS FAQ.TXT WHICH IS MY KNOWLEGE BASE
print("Loading knowledge base")
loader = TextLoader(r"C:\Users\Pujab\Documents\Getting started\faq.txt")# Ensure faq.txt is in the same folder or use an absolute path

# Test if the file can be opened
with open(r"C:\Users\Pujab\Documents\Getting started\faq.txt", "r") as file:
    print(file.read())  # Print the content of faq.txt to ensure it’s being read


print("Converting text into chunks.....")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(loader.load())

#step 3 Generate embeddings storen in FAISS
print("Generate embeddings...")
embeddings = OpenAIEmbeddings(api_key=api_key)
vectorstore = FAISS.from_documents(docs,embeddings)
print("Embeddings stored in FAISS")

#Step 4 setup the RetrivalQA pipeline
print(" Setting up RAG pipeline...")
llm = ChatOpenAI(api_key=api_key, model="gpt-4")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)


#RetrievalQA is the class that connects the language model (llm) to the vectorstore (vectorstore) enabling a retrieval-augmented generation (RAG) workflow.


#Step 5:Chatbot :oop for user queries
print(" RAG Chatbot is ready! Type 'exit' to quit.")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("Bye Bye Bye")
        break
    response = qa_chain.invoke({"query": query})
    print("Bot:", response["result"])  # Output the answer
    # Optional: Print the source documents if you want more context
    print("\nSources:")
    for doc in response["source_documents"]:
        print(f"- {doc.metadata['source']}")
