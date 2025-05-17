# This is for use in COLAB
%%capture
!pip install langchain pypdf openai chromadb tiktoken

------------
#to enter the CHATGPT data
from getpass import getpass
import os

OPENAI_API_KEY = getpass('Enter the secret value: ')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

--------
import requests
from langchain.document_loaders import PyPDFLoader

#When this example was used, these documents were newly created, so there was no time to review them.
#PDF files can also be uploaded.
urls = [
    'url/1.pdf',
    'url/2.pdf',
    'url/3.pdf',
    'url/4.pdf',
    'url/5.pdf'
]

ml_papers = []

for i, url in enumerate(urls):
    response = requests.get(url)
    filename = f'paper{i+1}.pdf'
    with open(filename, 'wb') as f:
        f.write(response.content)
        print(f'Descargado {filename}')
        loader=PyPDFLoader(filename)
        data=loader.load()
        ml_papers.extend(data)

# Use the ml_papers list to access the elements of all downloaded documents
print('Contenido:')
print()
----------------------------
#here is to see the length, type of documents and select the one in position 4
type(ml_papers), len(ml_papers), ml_papers[3]
------------
#Cut documents (split) to be able to embed
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,# text len
    chunk_overlap=200,#comparison letters
    length_function=len
    )

documents = text_splitter.split_documents(ml_papers)
------------------------------------
len(documents), documents[10]
------------------------------------
#Embeddings and send to vectorial DB
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
    )
--------------------------------------
#Chat templates and chains for information query

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type='stuff',
    retriever=retriever
)
-------------------------
#These are questions about data found in the documents
query = "qué es fingpt?"
qa_chain.run(query)

query = "qué hace complicado entrenar un modelo como el fingpt?"
qa_chain.run(query)

query = "qué es fast segment?"
qa_chain.run(query)

query = "cuál es la diferencia entre fast sam y mobile sam?"
qa_chain.run(query)
