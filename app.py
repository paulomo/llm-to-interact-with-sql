import os
import pinecone
from PyPDF2 import PdfReader

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINCONE_API_KEY = os.environ["PINCONE_API_KEY"]
APP_SECRET_KEY = os.environ["APP_SECRET_KEY"]

PINECONE_ENV = "gcp-starter"
from flask import Flask, flash, redirect, request

import langchain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.load.dump import dumps

# For SQL
from sqlalchemy import create_engine
from langchain.agents import AgentType, create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

app = Flask(__name__)
app.secret_key = APP_SECRET_KEY

prompt = ""
pinecone.init(api_key=PINCONE_API_KEY, environment=PINECONE_ENV)
pincone_index_name = "bchydrobot"
embeddings = OpenAIEmbeddings()
ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif"}


def calculate_similarity(query, response):
    vectorizer = TfidfVectorizer()
    tfidf_query = vectorizer.fit_transform([query])
    tfidf_response = vectorizer.transform([response])
    similarity = cosine_similarity(tfidf_query, tfidf_response)
    return similarity[0][0]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/ping", methods=["GET", "POST"])
def ping():
    return "App is running on port 5000"


@app.route("/add", methods=["GET", "POST"])
def add_doc():
    success = ""
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # print(pdf_file.filename)
            pdf_reader = PdfReader(file)

            # Read text from PDF
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()

            # splitting text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.create_documents(pdf_text)
            Pinecone.from_documents(chunks, embeddings, index_name=pincone_index_name)

    success = "Successful"

    return success


@app.route("/ask", methods=["GET", "POST"])
def ask_ai():
    query = request.json["question"]

    prompt_template = """
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:
    """

    PROMPT = langchain.PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    doc_search = Pinecone.from_existing_index(pincone_index_name, embedding=embeddings)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=doc_search.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )

    response = qa({"query": query})
    # print("Response : ", response["result"])
    print("Source Documents : ", response["source_documents"])

    return response["result"]


### ENVs
SQL_URI = os.environ["SQL_URI"]
db_engine = create_engine(SQL_URI)


@app.route("/chat-db", methods=["GET", "POST"])
def chat_db():
    query = request.json["question"]

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful AI assistant expert in querying SQL Database to find answers to user's question about Products and Orders.
                """,
            ),
            ("user", "{question}\n ai: "),
        ]
    )

    db = SQLDatabase(db_engine)
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_toolkit.get_tools()

    sqldb_agent = create_sql_agent(
        llm=llm,
        toolkit=sql_toolkit,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )

    response = sqldb_agent.run(final_prompt.format(question=query))

    return response


if __name__ == "__main__":
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
