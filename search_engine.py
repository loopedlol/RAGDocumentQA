import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import util

load_dotenv()

llm = ChatOpenAI(
    model = "gpt-4.1-mini",
    temperature = 0,
    max_completion_tokens = 400,
    max_retries = 2
)

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

if isinstance(query_embedding, list):
    query_embedding = np.array(query_embedding)

stored_embeddings_tensor = torch.tensor(stored_embeddings, dtype=torch.float32)
query_embedding_tensor = torch.tensor(query_embedding, dtype=torch.float32)

scores = util.dot_score(query_embedding_tensor.view(1, -1), stored_embeddings_tensor)[0].numpy()
highest_score_index = np.argmax(scores)
highest_score_passage = texts[highest_score_index]

print(f"Answer: {highest_score_passage}")

template = ChatPromptTemplate.from_messages(
    [
        ("system", "Use the passage given by the user to generate answers to the user's questions."),
        ("human", "The passage is as follows: \n{context} \n\nMy question is as follows: \n{question}")
    ]
)

prompt_value = template.invoke(
    {
        "context": highest_score_passage,
        "question": question1
    }
)

llm_message = llm.invoke(prompt_value)

print(llm_message.content)