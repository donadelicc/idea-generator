from flask import Flask, request, render_template
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os
import logging

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



SYSTEM_PROMPT = """
Mål: Ditt mål med denne applikasjonen er å hjelpe entreprenørskapsstudenter med å generere innovative
og kreative løsninger på spesifikke og nyanserte problemer. Brukerne av denne tjenesten vil presentere unike utfordringer,
og din oppgave er å svare med originale og gjennomførbare ideer som kan stimulere videre utvikling. \
Prioriteringer: \
Kreativitet og Innovasjon: Løsningene du foreslår skal være unike, nyskapende, og gjerne utradisjonelle.
Du skal prioritere ideer som går utenfor de vanlige rammeverkene og som kan inspirere til videre utforskning og testing.\
Relevans og Gjennomførbarhet: Selv om kreativitet er viktig, må løsningene dine også være relevante for problemstillingen
og ha et potensial for praktisk gjennomføring i en reell kontekst.\
Datainformert Innsikt: Du skal bruke innsikt fra et bredt spekter av datakilder og referanser for å styrke forslagene dine.
Dine løsninger skal bygge på kunnskap og trender fra ulike bransjer, teknologier, og samfunnsforhold.

Under har du er en mal på hva en syretest kan inneholde.
Det er ikke obligatorisk at alle punktene i Syretestmalen skal besvarer.
Du må gjøre en vurdering på hvilke aspekter som er mest sentrale å belyse i forhold til problemstillingen. \

Syretestmal: {context}

"""

loader = PyPDFLoader("./data/Syretestmal.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)


rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)




@app.route('/', methods=['GET', 'POST'])
def index():
    response_text = ""
    if request.method == 'POST':
        input = request.form['user_input']
        response_text = rag_chain.invoke(input)
        vectorstore.delete_collection()

    return render_template('index.html', response_text=response_text)


# if __name__ == '__main__':
#     app.run(debug=True)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
    