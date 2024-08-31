from flask import Flask, request, render_template
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
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
# model = AzureChatOpenAI(
#     openai_api_version="2024-02-15-preview",
#     azure_deployment="gpt-4o",
#     temperature=0.1
# )

app = Flask(__name__)

# LOGGING
logging.basicConfig(level=logging.INFO)
azure_logger = logging.getLogger('azure.core')
# Sett loggnivået til WARNING for å redusere mengden loggmeldinger
azure_logger.setLevel(logging.WARNING)

@app.route('/', methods=['GET', 'POST'])
def index():
    response_text = ""
    if request.method == 'POST':
        user_input = request.form['user_input']
        messages = [
            SystemMessage(content="Vår en snill chatbot"),
            HumanMessage(content=user_input),
        ]
        parser = StrOutputParser()
        chain = model | parser
        
        response_text = chain.invoke(messages)

    return render_template('index.html', response_text=response_text)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    