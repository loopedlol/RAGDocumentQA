import pandas as pd
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

df = pd.read_parquet("hf://datasets/LGAI-EXAONE/Ko-LongRAG/data/test-00000-of-00001.parquet")
passage1 = df["context"][0]

def count_words(text: str) -> int: #Not used currently because korean text is different
    return len(text.split())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 50,
    length_function = len
)
texts = text_splitter.split_text(passage1)

model = SentenceTransformer("intfloat/multilingual-e5-small")

editted_texts = []
for text in texts:
    editted_texts.append(f"passage: {text}")

embeddings = model.encode(editted_texts, show_progress_bar = True, normalize_embeddings = True)

np.save("test_embeddings.npy", embeddings)