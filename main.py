import fitz  # PyMuPDF
from fastapi import UploadFile


def extract_text_from_pdf(file_bytes: bytes):
    # Open the PDF file
    # Read the file contents into memory
    # file_bytes = file.file.read()

    # Open the PDF file from the bytes
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")

    # Extract text from each page
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()

    return text

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""Use the following pieces of context to answer the questions at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use five sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.

    <context>
    {context}
    </context>

    Question: {question}
    Helpful Answer, formatted in markdown:
    """,
    input_variables=["context", "question"],
)

from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import Chroma #import vectordb - a db to store and retrieve embeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


def process_text_variable(text_variable):
    # Save the text variable to a temporary file
    with open("temp_text.txt", "w") as f:
        f.write(text_variable)

    # Load the text file using TextLoader
    loader = TextLoader("temp_text.txt")
    documents = loader.load()
    # print(documents[0])
    print("document loaded!!!")
    return documents

def ragModel(text, question):
  if not text.strip():
      llm = ChatOpenAI(temperature=0.5, api_key=process.env.OPEN_AI_API_KEY)
      result = llm(question)
      return result.content

  documents = process_text_variable(text)

  #chunking the docs
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
  documents = text_splitter.split_documents(documents)

  #embedding the chunks
  embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

  #initializing the vector db
  db = Chroma.from_documents(
    documents = documents,
    embedding= embeddings_model,
    persist_directory="my_chroma_db_embeddings",
    )

  #extracting context from vector db
  context_docs = db.similarity_search(question)
  context_docs = "\n".join([doc.page_content for doc in context_docs])

  llm = ChatOpenAI(temperature=0.5)
  qa_chain = LLMChain(llm=llm, prompt=prompt)
  result = qa_chain(
    {"context": context_docs, "question": question}
  )
  return result['text']





from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from typing import Optional


#import requests
app = FastAPI()


@app.get('/')
async def home():
  return "Welcome to my OpenAI enabled RAG App"

@app.post("/upload/")
async def upload_file(
    context_file: UploadFile | None = None,
    text: str | None = Form(None)
):
    response = {}
    pdf_text = ""
    
    if context_file:
        response["file_name"] = context_file.filename
        response["file_content_type"] = context_file.content_type
    if text:
        response["text"] = text

    if context_file:
        content = await context_file.read()
        extracted_text = extract_text_from_pdf(content)
        pdf_text += extracted_text

    model_response = ragModel(pdf_text, text)
    response['model_response'] = model_response

    return response
  
