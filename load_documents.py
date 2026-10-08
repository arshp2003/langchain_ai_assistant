from langchain_community.document_loaders import PyPDFLoader


pdf_path = "data/documents/LangChain_Notes.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("Number of pages:", len(documents))

for document in documents[:2]:
    print("\n--- PAGE ---")
    print(document.page_content[:1000])