from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
You are a question-answering system.

Answer the question using only the provided passages.

Rules:
- Give the shortest answer that fully answers the question.
- If the answer is a name, place, date, year, number, title, or short phrase, output only that answer.
- Do not include extra explanation unless it is necessary.
- Do not use outside knowledge.
- If the passages do not contain the answer, respond exactly: NOT_FOUND
- If multiple passages conflict, use the passage that most directly answers the question.
"""

class Engine():

    def search_chroma(self, query: str, persist_directory: str, k: int):
        vector_store = Chroma(
            embedding_function = embedding_model,
            persist_directory = persist_directory
        )

        results = vector_store.similarity_search(f"query: {query}", k)
        return results
    
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