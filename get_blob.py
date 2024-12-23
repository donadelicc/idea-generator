from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import PyPDFLoader
import tempfile

import os
from dotenv import load_dotenv

load_dotenv()

def load_pdf_from_blob():
   """
   Retrieves and loads a PDF file from Azure Blob Storage into document format.
   
   Returns:
       List[Document]: The loaded PDF documents
       
   Raises:
       ResourceNotFound: If the blob or container doesn't exist
       Exception: For other Azure storage related errors
   """
   try:
       connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
       container_name = os.getenv('CONTAINER_NAME')
       blob_name = 'Syretestmal.pdf'
       
       # Create the BlobServiceClient and get blob content
       blob_service_client = BlobServiceClient.from_connection_string(connect_str)
       container_client = blob_service_client.get_container_client(container_name)
       blob_client = container_client.get_blob_client(blob_name)
       blob_data = blob_client.download_blob()
       pdf_content = blob_data.readall()
       
       # Save to temporary file and load PDF
       with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
           temp_file.write(pdf_content)
           temp_file_path = temp_file.name
       
       loader = PyPDFLoader(temp_file_path)
       docs = loader.load()
       
       # Clean up temporary file
       os.unlink(temp_file_path)
       return docs
       
   except Exception as e:
       raise Exception(f"Error retrieving PDF from blob storage: {str(e)}")