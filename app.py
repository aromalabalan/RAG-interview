import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def load_resume():
    loader=PyPDFLoader("resume.pdf")

    documents = loader.load()

    return documents

def doc_splitter(documents):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap=100)

    chunks = text_splitter.split_documents(documents)

    return chunks

def embedding():
    embedding=OpenAIEmbeddings(model ="text-embedding-3-small")

    return embedding

def vector_db(chunks, embedding):
    vec_db = FAISS.from_documents(documents=chunks,embedding=embedding
    )
    return vec_db

def load_llm():

    llm=ChatOpenAI(model="gpt-4o-mini",temperature=0.2)

    return llm

prompt= ChatPromptTemplate.from_template(""" you are an AI assistant answering que based on the document provided
use that provided context only. DO NOT MAKE UP ANY INFORMATION FROM YOUR KNOWLEDGE. If the ans is not available in the document context, say like 'it is not mentioned in the document
context:{context}

question : {question}
Answer:""")

def ans(question, llm, vec_db):

    docs = vec_db.similarity_search(question, k=5)

    if not docs:
        return "No relevant information found."

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    msg = prompt.format_messages(context=context,question=question)

    response = llm.invoke(msg)

    return response.content

def main():
    document = load_resume()

    chunks = doc_splitter(document)

    embeddings = embedding()

    vect__DB=vector_db(chunks, embeddings)

    llm=load_llm()

    while True:
        question = input("\n You:   ")

        if question.lower() in ["exit", "quit"]:
            break

        answer=ans(question,llm, vect__DB)

        print("Ans:", answer)

if __name__=="__main__":
    main()