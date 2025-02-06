from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def setup_vectorstore(documents, persist_directory="./data/chroma_db"):
    """Converts documents into embeddings and stores them in a vector database."""
    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Generate embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create Chroma vector store
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

    return vectorstore
