from flask import Flask, request, render_template
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import LLMChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from typing_extensions import List, TypedDict


from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os
import logging

from PROMPT import SYSTEM_PROMPT, SYRESTESTMAL

load_dotenv()


embeddings = AzureOpenAIEmbeddings(

    openai_api_version="2023-07-01-preview",
    azure_deployment="text-embedding-ada-002",
)

model = AzureChatOpenAI(
    api_version="2023-07-01-preview",
    azure_deployment="gpt-4",
)

app = Flask(__name__)

# LOGGING
logging.basicConfig(level=logging.INFO)
azure_logger = logging.getLogger('azure.core')
# Sett loggnivået til WARNING for å redusere mengden loggmeldinger
azure_logger.setLevel(logging.WARNING)


loader = PyPDFLoader("./data/Syretestmal.pdf")
docs = loader.load()

vector_store = InMemoryVectorStore(embeddings)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

_ = vector_store.add_documents(documents=all_splits)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = model.invoke(messages)
    return {"answer": response.content}


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


@app.route('/', methods=['GET', 'POST'])
def index():
    response_text = ""
    if request.method == 'POST':
        question = request.form['user_input']
        response_text = graph.invoke({"question": question})["answer"]
        #vectorstore.delete_collection()
        # vectorstore.reset_collection()

    return render_template('index.html', response_text=response_text)


if __name__ == '__main__':
    app.run(debug=True)
    

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000, debug=True)
    