from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap = 100,
    length_function = len
)

embedding_model = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-small",
    encode_kwargs = {"normalize_embeddings": True}
)

class Embedder():
    
    def generate_embeddings(self, context: str, persist_directory: str) -> Chroma:
        raw_chunks = text_splitter.split_text(context)

        edited_chunks = []
        for i, chunk in enumerate(raw_chunks):
            edited_chunks.append(Document(page_content = f"passage: {chunk}", metadata = {"chunk_index": i}))
        
        return Chroma.from_documents(
            documents = edited_chunks,
            embedding = embedding_model,
            persist_directory = persist_directory
        )

