from chromadb.utils import embedding_functions
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):

    def __init__(self) -> None:
        self._embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedding_function(input=texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedding_function(input=[text])[0]


embeddings = LocalEmbeddings()