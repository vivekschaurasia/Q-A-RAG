import os
import faiss
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv


# Step 1: Load and Preprocess News Articles
folder_path = "news_data path"

documents = [
    open(os.path.join(folder_path, filename), 'r', encoding='utf-8').read()
    for filename in os.listdir(folder_path) if filename.endswith(".txt")
]

# Filter out failed retrieval messages
news_documents = [doc for doc in documents if doc != "Failed to retrieve the webpage."]

# Step 2: Split Documents into Chunks
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)

all_chunks = [chunk for doc in news_documents for chunk in text_splitter.split_text(doc)]

# Step 3: Load Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key="Your API Key")

# Generate embeddings for each chunk
chunk_embeddings = embeddings.embed_documents(all_chunks)


embedding_dim = len(chunk_embeddings[0]) 

# Convert chunk_embeddings to a numpy array with float32 type
chunk_embeddings_np = np.array(chunk_embeddings).astype("float32")


# Step 4: Initialize FAISS Index
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance is used; you can choose other metrics
index.add(chunk_embeddings_np)  # Add embeddings to the FAISS index

"""
# Step 5: Define the FAISS-Based Retriever
def retrieve_similar_chunks(query, k=3):
    #query_embedding = np.array([embeddings.embed_documents(query)]).astype("float32")
    query_embedding = np.array([embeddings.embed_query(query)]).astype("float32")

    distances, indices = index.search(query_embedding, k)
    return [all_chunks[i] for i in indices[0] if distances[0][i] < 0.1]  # Using a threshold of 0.1
"""


# Step 5: Define the FAISS-Based Retriever
def retrieve_similar_chunks(query, k=5):
    # Embed the query and ensure it has the correct shape for FAISS
    query_embedding = np.array(embeddings.embed_documents([query])).astype("float32")
    
    # Check if query_embedding has shape (1, embedding_dim)
    if query_embedding.shape != (1, embedding_dim):
        query_embedding = query_embedding.reshape(1, embedding_dim)
    
    # The search
    distances, indices = index.search(query_embedding, k)
    
    # Filter results 
    relevant_chunks = [
        all_chunks[i] for i in indices[0] 
        if i < len(all_chunks) and distances[0][indices[0].tolist().index(i)] < 0.7
    ]
    
    return relevant_chunks



# Step 6: Define the QA Function
def answer_question(query):
    # Retrieve relevant documents
    relevant_docs = retrieve_similar_chunks(query)

    # Combine the query with relevant document content
    combined_input = (
        "Answer the following question based only on the provided documents: "
        + query
        + "\n\nRelevant Documents:\n"
        + "\n\n".join(relevant_docs)
        + "\n\nIf the answer is not found in the documents, respond with 'I'm not sure'."
    )

    
    messages = [
        SystemMessage(content="You are a helpful assistant for answering questions based on news documents."),
        HumanMessage(content=combined_input),
    ]

    # ChatOpenAI model to generate the answer
    result = llm.invoke(messages)
    
    return result.content

# Step 7: Create the LLM Model
llm = ChatOpenAI(model="gpt-4o", api_key="Your API Key")

# Step 8: Test the QA Bot with a Sample Query
query = input("What is your query? ").lower()

while query != "exit":
    answer = answer_question(query)
    # Display the answer
    print("\n--- Answer ---")
    print(answer)
    query = input("What is your query? ").lower()