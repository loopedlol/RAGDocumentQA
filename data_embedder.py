import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 80,
    length_function = len
)

embedding_model = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-small",
    encode_kwargs = {"normalize_embeddings": True}
)

def split_articles(text: str):
    articles = re.split(r"(?=Title:)", text)

    for i, article in enumerate(articles):
        articles[i] = article.strip()
    return articles


class Embedder():
    
    def generate_embeddings(self, context: str, persist_directory: str) -> Chroma:
        
        articles = split_articles(context)
        edited_chunks = []

        global_chunk_index = 0

        for article_index, article in enumerate(articles):
            raw_chunks = text_splitter.split_text(article)

            for article_chunk_index, chunk in enumerate(raw_chunks):
                edited_chunks.append(
                    Document(
                        page_content=f"passage: {chunk}",
                        metadata={
                            "chunk_index": global_chunk_index,
                            "article_index": article_index,
                            "article_chunk_index": article_chunk_index
                        }
                    )
                )

                global_chunk_index += 1
        
        return Chroma.from_documents(
            documents=edited_chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )