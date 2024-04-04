from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl



custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    """
    Generate a retrieval-based question answering chain.

    Args:
        llm (LLM): The language model used for generating queries.
        prompt (str): The prompt or initial context for the question answering chain.
        db (Database): The database used for retrieving relevant documents.

    Returns:
        RetrievalQA: The retrieval-based question answering chain.
    """
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    """
    Load the LLM model.

    Returns:
        CTransformers object: The loaded model.
    """
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "C:/Users/User/chatbot/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm
DB_FAISS_PATH = "C:/Users/User/chatbot/vectorstore\db_faiss"
#QA Model Function
def qa_bot():
    """
    Generates a QA bot by initializing the necessary components and returning the QA object.

    Returns:
        qa (retrieval_qa_chain): The QA object used for question answering.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': 'cuda'})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response
#Call the final_result function with your query
query = "What are symptom of flu?"
answer = final_result(query)

# Print or display the answer
print(answer)

#chainlit code
@cl.on_chat_start
async def start():
    """
    Initializes and starts the chatbot.

    This function is decorated with the `cl.on_chat_start` decorator, which means it will be triggered when a chat session starts.
    """
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    """
    Asynchronous function that handles a message received by the client.

    Parameters:
        message (object): The message object received by the client.

    Returns:
        None
    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    message_text = message.content  # Extract the text content from the message object

    res = await chain.ainvoke({'query': message_text}, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        source_info = []
        for doc in sources:
            # Attempt to access title from a dictionary or object attribute
            if hasattr(doc, 'title'):  # If 'title' is an attribute of the object
                source_info.append(doc.title)
            elif 'title' in doc:  # If 'title' is a key in a dictionary
                source_info.append(doc['title'])
            else:
                source_info.append("A relevant document")  # Fallback text if title is not accessible
        answer += "\nSources: " + ', '.join(source_info)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()

