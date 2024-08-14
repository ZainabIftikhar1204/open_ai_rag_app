# OpenAI RAG App
This is a RAG (Retrieval Augmented Generation) App. This means that you can utilize the app to ask questions about a particular document that you have. The app takes a PDF file as an optional argument so you can ask any question related to the content in that particular document. Otherwise, you can also ask any generic question from the application without uploading any document. 

## Interface:
You can find a simple interface developed with gradio on the following link:
[OpenAI RAG App Interface](https://zainab1204-openai-rag.hf.space/)

#### How to use
The use is similar to the description. You can either ask a generic question from the app by typing in your query in the **question** field or you can upload a PDF file of any size and then ask a question relevant to that document. 


## Public Endpoints:
The application is also available through public endpoints created with the help of FastAPI. You can access the deployed endpoints here:
[FastAPI Endpoints for OpenAI RAG App](https://open-ai-rag-app.onrender.com/docs)

#### How to use
The above link takes you to a simple UI that you can use to check the **/upload** endpoint. You can use it in the way similarly by giving it just a generic query or uploading a PDF file and then asking a question specific to that file.

## Run with Docker
You can also create a docker image of the application by the following steps:
1. Clone the repo to your system
2. Create a .env file in the project and add your OpenAI access key as OPENAI_API_KEY
3. Run ```docker-compose up --build``` in your terminal
