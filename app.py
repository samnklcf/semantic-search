from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import os

# Créer une instance FastAPI
app = FastAPI()

# Configuration de l'environnement pour l'API Google
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCBjcpOQAYX53GtzBpDXRvfoMbTp19SjxM"

# Embeddings et variables globales
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
directory = './content/pets'
persist_directory = "chroma_db"

# Charger les documents
def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

# Splitter les documents
def split_docs(documents, chunk_size=2000, chunk_overlap=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    
    return docs

# Initialiser la base de données de vecteurs
def init_vector_db():
    documents = load_docs(directory)
    docs = split_docs(documents)
    db = Chroma.from_documents(docs, embeddings)
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return db

# Initialisation du LLM et du chain de question-réponse
def init_qa_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    return load_qa_chain(llm, chain_type="stuff", verbose=True)

# Initialiser la base de vecteurs et la chaîne QA
vector_db = init_vector_db()
qa_chain = init_qa_chain()

# Modèle de données pour la requête utilisateur
class QueryRequest(BaseModel):
    query: str

# Endpoint principal pour les requêtes de question-réponse
@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        # Recherche de similarité dans la base de données vectorielle
        matching_docs = vector_db.similarity_search(request.query)
        print(matching_docs)

        # Exécution de la chaîne de question-réponse
        answer = qa_chain.run(input_documents=matching_docs, question=request.query)

        return {"query": request.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Lancement du serveur avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)