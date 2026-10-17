from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from embeddings import embeddings


pdf_path = "data/documents/LangChain_Notes.pdf"
vector_db_path = "chroma_db"
collection_name = "langchain_notes"


# 1. Load PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("Pages:", len(documents))


# 2. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Chunks:", len(chunks))


# 3. Open Chroma database
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=vector_db_path
)


# 4. Reset collection so re-runs don't duplicate chunks
vector_store.reset_collection()


# 5. Embed and store chunks
vector_store.add_documents(chunks)

print(f"Documents embedded and stored in Chroma collection '{collection_name}'.")


# 6. Confirm database was saved locally
print(f"Chroma database saved at: {vector_db_path}")
print("Vectors stored:", vector_store._collection.count())