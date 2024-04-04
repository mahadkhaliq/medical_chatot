# Import necessary modules from the langchain library database creation
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define the path to the data directory and the path to save the vector database
DATA_PATH = '/content/drive/MyDrive/NLP/'  # Update this to the directory containing your PDF files
DB_FAISS_PATH = '/content/drive/MyDrive/NLP/vectorstore/db_faiss'

# Define a function to create a vector database
"""
This is a Python script that creates a vector database using the langchain library. The script imports several modules from the langchain library, including HuggingFaceEmbeddings, FAISS, PyPDFLoader, DirectoryLoader, and RecursiveCharacterTextSplitter.

The script defines a function called create_vector_db() that creates a vector database. The function does the following:

It creates an instance of the DirectoryLoader class, which loads all PDF files from the specified data path using the PyPDFLoader class.
It loads the documents and splits them into smaller chunks using an instance of the RecursiveCharacterTextSplitter class.
It creates an instance of the HuggingFaceEmbeddings class, which is used to generate embeddings for the text chunks.
It creates an instance of the FAISS class, which is used to create a vector database from the text chunks and their embeddings.
It saves the vector database to a local file specified by the DB_FAISS_PATH variable.
"""
def create_vector_db():
    # Create an instance of the DirectoryLoader class to load PDF files from the data directory
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    # Load the documents and split them into smaller chunks using an instance of the RecursiveCharacterTextSplitter class
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create an instance of the HuggingFaceEmbeddings class to generate embeddings for the text chunks
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Create an instance of the FAISS class to create a vector database from the text chunks and their embeddings
    db = FAISS.from_documents(texts, embeddings)

    # Save the vector database to a local file
    db.save_local(DB_FAISS_PATH)

# Check if the script is being run as the main program and call the create_vector_db() function if it is
if __name__ == "__main__":
    create_vector_db()

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
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm
DB_FAISS_PATH = '/content/drive/MyDrive/NLP/vectorstore/db_faiss'
#QA Model Function
def qa_bot():
    """
    Generates a QA bot by initializing the necessary components and returning the QA object.

    Returns:
        qa (retrieval_qa_chain): The QA object used for question answering.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': 'cpu'})

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
# Call the final_result function with your query
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
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()