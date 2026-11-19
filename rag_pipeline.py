
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os


def build_qa_chain():
    """
    Builds and returns the Conversational RAG QA chain.
    Uses an existing ChromaDB database instead of
    recreating embeddings every time the application runs.
    """

    # 1. Locate the project directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 2. Check whether NCERT text file exists
    text_file = os.path.join(BASE_DIR, "ncert_text.txt")

    if not os.path.exists(text_file):
        raise FileNotFoundError(
            "NCERT text file not found. Please check ncert_text.txt."
        )

    # 3. Connect to the embedding model
    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # 4. Load the existing ChromaDB database
    db_path = os.path.join(BASE_DIR, "ncert_db_local")

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            "ChromaDB database not found. Please create ncert_db_local first."
        )

    vectorstore = Chroma(
        collection_name="langchain",
        embedding_function=embedding_model,
        persist_directory=db_path
    )

    # 5. Create a retriever
    retriever = vectorstore.as_retriever()

    # 6. Connect to the Phi-3 language model
    llm = ChatOllama(
        model="phi3-fast"
    )

    # 7. Define the RAG prompt template
    prompt_template = ChatPromptTemplate.from_template("""
    You are a helpful and personalized AI tutor for a Class 10 student.

    Rules for answering:
    - Always answer in **English only**.
    - Do NOT add phrases like "according to the study material",
      "the text says", or "from the PDF".
    - Use simple, clear, step-by-step explanations.
    - If the student says "I didn't understand",
      simplify with everyday examples.
    - If the student says "oh got it", "good job", "thanks",
      "oh nice", or any similar phrase, reply briefly with
      encouragement (e.g., "Great! Happy you understood 🎉").
    - If the student gives feedback but NOT a question,
      DO NOT explain again — just acknowledge warmly.
    - Only give detailed explanations if the student
      actually asks a question.
    - Be concise and avoid repeating long explanations
      unless specifically asked.
    - Focus on the student's latest input, while remembering
      the conversation history.
    - Base your explanation strictly on the
      "Study material context" below.
    - Do NOT add outside knowledge unless the student
      explicitly asks for it.

    Conversation so far:
    {chat_history}

    Study material context:
    {context}

    Student's latest message: {question}

    AI Tutor (in English):
    """)

    # 8. Setup conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 9. Build the Conversational RAG chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt_template
        }
    )

    return qa_chain


# 10. Run the chatbot directly from Terminal
if __name__ == "__main__":

    qa_chain = build_qa_chain()

    print("\nAI Tutor is ready!")
    print("Ask a question about your Class 10 NCERT book.")
    print("Type 'exit' to quit.\n")

    while True:
        user_question = input("Student: ")

        if user_question.lower() == "exit":
            print("Goodbye!")
            break

        response = qa_chain.invoke({
            "question": user_question
        })

        print("\nAI Tutor:", response["answer"])
        print()
