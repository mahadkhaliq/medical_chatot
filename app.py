from flask import Flask, request, jsonify
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

model_path = "./llama-2-7b-chat.ggmlv3.q8_0.bin"
db_faiss_path = "./vectorstore/db_faiss"
app = Flask(__name__)

# Custom prompt template for the QA retrieval
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Function to set the custom prompt
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Function to load the LLM model
def load_llm():
    llm = CTransformers(
        model=model_path,  # Adjust this path to your model's location
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# Global initialization to load components once
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': 'cuda'})
db = FAISS.load_local(db_faiss_path, embeddings, allow_dangerous_deserialization=True)  # Adjust path accordingly
llm = load_llm()
qa_prompt = set_custom_prompt()
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': qa_prompt}
                                       )

# The API endpoint for performing queries
@app.route('/query', methods=['POST'])  # This line is crucial
def query():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing query'}), 400

    query_text = data['query']
    context = data.get('context', '')  # Optional context field

    # Using the preloaded QA chain to get the answer
    response = qa_chain({'context': context, 'query': query_text})

    # Serialize the response here, modifying as necessary to match your response structure
    if hasattr(response, 'documents'):
        documents = response.documents
        serialized_docs = [{'title': doc.title, 'content': doc.content} for doc in documents]  # Customize based on your Document object attributes
        return jsonify({'answer': serialized_docs}), 200
    else:
        # Handle other types of responses, or adjust as per your response structure
        return jsonify({'answer': str(response)}), 200

if __name__ == '__main__':
    app.run(debug=True)
