from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-small",
    encode_kwargs = {"normalize_embeddings": True}
)

llm = ChatOpenAI(
    model = "gpt-4.1-mini",
    temperature = 0,
    max_completion_tokens = 400,
    max_retries = 2
)

ANSWER_SYSTEM_PROMPT = """
You are a strict question-answering system.

Answer the question using only the provided passages.

Rules:
- Give the shortest answer that fully answers the question.
- Article titles, category labels, lists, and metadata are valid evidence.
- If the answer is a name, place, date, year, number, title, or short phrase, output only that answer.
- If the question asks for a difference, comparison, reason, or explanation, answer in one concise sentence.
- Do not use outside knowledge.
- If the passages contain multiple possible answers, choose the one that directly matches the wording of the question.
- If the answer truly cannot be found in the passages, respond exactly: NOT_FOUND.
"""

class Engine():

    def search_chroma(self, query: str, persist_directory: str, k: int):

        vector_store = Chroma(
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )

        results = vector_store.similarity_search(f"query: {query}", k)

        all_passages = vector_store.get()

        edited_results = []
        added_indexes = []

        for i in range(0, len(results), 1):
            doc_index = results[i].metadata["chunk_index"]

            wanted_indexes = [
                doc_index - 1,
                doc_index,
                doc_index + 1
            ]

            for document_text, metadata in zip(all_passages["documents"], all_passages["metadatas"]):
                chunk_index = metadata["chunk_index"]

                if (chunk_index in wanted_indexes) and (chunk_index not in added_indexes):
                    edited_results.append(
                        Document(
                            page_content=document_text,
                            metadata=metadata
                        )
                    )
                    added_indexes.append(chunk_index)

        edited_results.sort(key=lambda document: document.metadata["chunk_index"])

        return edited_results
    
    
    def query_llm(self, query: str, context: str) -> str:
        template = ChatPromptTemplate.from_messages(
            [
                ("system", ANSWER_SYSTEM_PROMPT),
                ("human", "The passage is as follows: \n{context} \n\nThe question is as follows: \n{question}")
            ]
        )

        prompt_value = template.invoke(
            {
                "context": context,
                "question": query
            }
        )

        llm_message = llm.invoke(prompt_value)
        return str(llm_message.content)