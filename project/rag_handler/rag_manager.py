# Created by Mohammed Jassim at 30/01/25
from langchain.document_loaders import TextLoader

# Document retrieval
loader = TextLoader("project/my_wikipedia_docs.txt")
documents = loader.load()

# Split Document
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Embedding and vector
# Sentence Transformers
from langchain.embeddings import HuggingFaceEmbeddings

from huggingface_hub import snapshot_download

# Download model to a local directory
model_path = snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="./models/all-MiniLM-L6-v2"
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# local vector store (FAISS)
from langchain.vectorstores import FAISS

vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("my_faiss_index")  # Save for reuse

# Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 chunks

# Local LLM for Generation
# Use a small, CPU-friendly model like GPT-2 or DistilBERT
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")


# Combine RAG Components
def rag_response(query):
    # Retrieve relevant chunks
    docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"

    # Generate answer with proper parameters
    answer = generator(
        prompt,
        max_new_tokens=200,  # Generate up to 100 new tokens
        truncation=True,  # Explicit truncation
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id  # Explicit padding
    )

    return answer[0]['generated_text']


if __name__ == '__main__':

    # query = "What causes auroras?"
    query = "What is the color of orange?"
    print(rag_response(query))