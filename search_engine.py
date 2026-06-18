from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

class Engine():

    def search_chroma(self, query: str, persist_directory: str, k: int):
        vector_store = Chroma(
            embedding_function = embedding_model,
            persist_directory = persist_directory
        )

        results = vector_store.similarity_search(f"query: {query}", k)
        return results
    
    def query_llm(self, query: str, context: str):
        template = ChatPromptTemplate.from_messages(
            [
                ("system", "Answer the question using only the provided passage. If the passage does not contain the answer, say that the answer is not found in the passage."),
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
        return llm_message