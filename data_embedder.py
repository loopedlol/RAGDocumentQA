import pandas as pd
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

df = pd.read_parquet("hf://datasets/LGAI-EXAONE/Ko-LongRAG/data/test-00000-of-00001.parquet")
passage1 = df["context"][0]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 50,
    length_function = len
)
texts = text_splitter.split_text(passage1)

model = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-small",
    encode_kwargs = {"normalize_embeddings": True}
)

editted_texts = []
for text in texts:
    editted_texts.append(f"passage: {text}")

document_embeddings = model.embed_documents(editted_texts)
document_embeddings_array = np.array(document_embeddings, dtype = np.float32)

np.save("test_embeddings.npy", document_embeddings_array)