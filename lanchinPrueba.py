# esto es para usar en COLAB
%%capture
!pip install langchain pypdf openai chromadb tiktoken

------------
#para ingresar los datos de CHATGPT
from getpass import getpass
import os

OPENAI_API_KEY = getpass('Enter the secret value: ')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

--------
import requests
from langchain.document_loaders import PyPDFLoader

#cuando se uso este ejemplo estos documentos estaban recien montados
#por lo cual no le dio tiempo a revisarlos
#tambien se pueden colocar archivos pdf
urls = [
    'https://arxiv.org/pdf/2306.06031v1.pdf',
    'https://arxiv.org/pdf/2306.12156v1.pdf',
    'https://arxiv.org/pdf/2306.14289v1.pdf',
    'https://arxiv.org/pdf/2305.10973v1.pdf',
    'https://arxiv.org/pdf/2306.13643v1.pdf'
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

# Utiliza la lista ml_papers para acceder a los elementos de todos 
#los documentos descargados
print('Contenido de ml_papers:')
print()
----------------------------
#aqui es para ver la longitud, tipo de documentos y seleccionar 
#el de la posicion 4
type(ml_papers), len(ml_papers), ml_papers[3]
------------
------------
#Cortar documentos(split) para poder hacer embebbing
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,#tamaño del text
    chunk_overlap=200,#caracteres de comparacion
    length_function=len
    )

documents = text_splitter.split_documents(ml_papers)
------------------------------------
len(documents), documents[10]
------------------------------------
#Embeddings e ingesta a base de datos vectorial
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}#buscamos los 3 fragmentos que mas se parecen al query
    #a mayor valor mayor gasto de token y talvez no quepa en el modelo
    )
--------------------------------------
#Modelos de chat y cadenas para consulta de información

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
#estas son preguntas de datos encontrados en los documentos
query = "qué es fingpt?"
qa_chain.run(query)

query = "qué hace complicado entrenar un modelo como el fingpt?"
qa_chain.run(query)

query = "qué es fast segment?"
qa_chain.run(query)

query = "cuál es la diferencia entre fast sam y mobile sam?"
qa_chain.run(query)

Primero veria en que esta fallando, esta dando respuestas incoherentes? entonces esta alucinando y bajaria la temperatura a respuestas concretas. temperature=0.0
es un chat que solo se basa en dar respuestas o un chatbot que guarda las interacciones anteriores?si es el primero reviso si usa embebbing y si el modelo es de texto
Revisaria si esta usando base de datos vectorial

First, I'd see what's wrong. Is he giving incoherent answers? Then he's hallucinating and I'd lower the temperature to concrete answers. Temperature = 0.0
Is it a chat that only focuses on providing answers or a chatbot that saves previous interactions? If it's the former, I check if it uses embedding and if the model is text-based.
I would check if there are using a vector database.But I need to know what error it gives and the use of the chat. Since a normal chat doesn't store previous responses, a chatbot does. It can also be created to search and analyze private information, among many other features.

If there hallucinating the answers, I'd set the temperature to "0". temperature=0.0. I'd also add a chain_type="stuff" , which is to take everything from the prompt.


comprobaria el output/max_tokens por la informacion incompleta y por el bloqueo comprobaria la interaccion, si el chat llama a todas las interacciones ya que al mayor cantidad de interacciones mas pesado, aqui es mejor hacer un resumen de las interacciones anteriores y solo colocar simpre las 5 ultimas.
tambien 

To reduce token usage I would limit input and output and to increase speed I would test different models and optimize memory.
ultimate_chain=SequentialChain(chains=[chain1, chain2],input_variables=["text","stile"],output,variables=["final_text"])
ultimate_chain({"text": input_text,"stile":"French"})