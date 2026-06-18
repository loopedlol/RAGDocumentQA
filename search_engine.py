import pandas as pd
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#from sentence_transformers import SentenceTransformer, util

df = pd.read_parquet("hf://datasets/LGAI-EXAONE/Ko-LongRAG/data/test-00000-of-00001.parquet")
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
    encode_kwargs = {"normalize_embeddings": True, "show_progress_bar": True}
)
#query_embedding = model.encode(f"query: {question1}", normalize_embeddings = True, show_progress_bar = True)

scores = util.dot_score(query_embedding, stored_embeddings)[0].numpy()
highest_score_index = np.argmax(scores)
highest_score = scores[highest_score_index]

print(f"Answer: {texts[highest_score_index]}")