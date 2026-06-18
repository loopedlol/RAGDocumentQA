import pandas as pd
import numpy as np
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_openai
from sentence_transformers import util

df = pd.read_parquet("hf://datasets/LGAI-EXAONE/Ko-LongRAG/data/test-00000-of-00001.parquet")
prompt = df["prompt"][0]
passage1 = df["context"][0]
question1 = df["question"][0]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 50,
    length_function = len
)
texts = text_splitter.split_text(passage1)

stored_embeddings = np.load("test_embeddings.npy")

model = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-small",
    encode_kwargs = {"normalize_embeddings": True}
)
query_embedding = model.embed_query(f"query: {question1}")
#query_embedding = model.encode(f"query: {question1}", normalize_embeddings = True, show_progress_bar = True)

if isinstance(query_embedding, list):
    query_embedding = np.array(query_embedding)

stored_embeddings_tensor = torch.tensor(stored_embeddings, dtype=torch.float32)
query_embedding_tensor = torch.tensor(query_embedding, dtype=torch.float32)

scores = util.dot_score(query_embedding_tensor.view(1, -1), stored_embeddings_tensor)[0].numpy()
highest_score_index = np.argmax(scores)
highest_score = scores[highest_score_index]

print(f"Answer: {texts[highest_score_index]}")