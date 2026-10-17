from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_path = "data/documents/Langchain_Notes.pdf"


# 1. Load PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()


# 2. Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# 3. Split documents
chunks = text_splitter.split_documents(documents)


print("Pages:", len(documents))
print("Chunks:", len(chunks))


# 4. Display first few chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)