
import pandas as pd
from data_embedder import Embedder
from search_engine import Engine

k = 1

def main():
    df = pd.read_parquet("hf://datasets/LGAI-EXAONE/Ko-LongRAG/data/test-00000-of-00001.parquet")
    embedder = Embedder()
    engine = Engine()

    for i in range(0, 10, 1):
        print("~" * 80)
        print(f"TEST #{i}")

        persist_directory = f"chroma_dbs/test_{i}"
        context = df["context"][i]
        query = df["question"][i]
        answer = df["answer"][i]

        embedder.generate_embeddings(context, persist_directory)
        results = engine.search_chroma(query, persist_directory, k)
        response = engine.query_llm(query, results[0].page_content)
        print(f"The AI Responsed with: {response.content}")
        print(f"The correct answer was: {answer}")

if __name__ == "__main__":
    main()