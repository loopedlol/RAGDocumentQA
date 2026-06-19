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

Compare the generated answer to the actual answer, using the question for context.

Grade CORRECT if:
- the generated answer directly answers the question, and
- it contains the same factual answer as the actual answer, even if wording is different, longer, or paraphrased.

Grade INCORRECT if:
- it gives the wrong entity, person, title, date, number, place, year, reason, or relationship
- it contradicts the actual answer
- it says the answer is not found when the actual answer is given
- it is too vague to verify
- it answers a different question

Important:
- If the actual answer is a short phrase and the generated answer includes that phrase with harmless extra context, grade CORRECT.
- Do not mark an answer incorrect only because it is longer than the actual answer.
- Be strict about factual identity.

Return exactly:
Grade: CORRECT or INCORRECT
Reason: one short sentence
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


def check_llm(question: str, llm_answer:str, actual_answer: str) -> str:
    template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "The original question is as follows: \n{question} \n\nThe generated answer is as follows: \n{llm_answer} \n\nThe actual answer is as follows: \n{actual_answer}")
        ]
    )

    prompt_value = template.invoke(
        {
            "question": question,
            "llm_answer": llm_answer,
            "actual_answer": actual_answer
        }
    )

    llm_message = llm.invoke(prompt_value)
    return str(llm_message.content)