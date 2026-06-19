from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import util

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

SYSTEM_PROMPT = """
You are grading a question-answering system.

Your job is to compare the generated answer with the actual answer.

Mark the generated answer as CORRECT if it gives the same meaning as the actual answer, even if:
- the wording is different
- the generated answer is longer
- the generated answer includes extra explanation
- the generated answer uses a translated or paraphrased form

Mark the generated answer as INCORRECT if:
- it contradicts the actual answer
- it gives a different entity, date, number, place, name, or reason
- it is too vague to verify
- it does not answer the question
- it only discusses a related topic without giving the actual answer

Ignore minor grammar differences, spacing differences, and punctuation differences.

Be strict with factual meaning. Extra information is okay only if it does not change or contradict the actual answer.

Return your judgment in this exact format:

Grade: CORRECT or INCORRECT
Reason: one short sentence explaining why
"""

def check_similarity(llm_answer: str, actual_answer: str):
    embeddings = embedding_model.embed_documents(
        [
            f"passage: {llm_answer}",
            f"passage: {actual_answer}"
        ]
    )

    score = util.dot_score(embeddings[0], embeddings[1]).item()
    return score


def check_llm(llm_answer:str, actual_answer: str) -> str:
    template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "The generated answer is as follows: \n{llm_answer} \n\nThe actual answer is as follows: \n{actual_answer}")
        ]
    )

    prompt_value = template.invoke(
        {
            "llm_answer": llm_answer,
            "actual_answer": actual_answer
        }
    )

    llm_message = llm.invoke(prompt_value)
    return str(llm_message.content)