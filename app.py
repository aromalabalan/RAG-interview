import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharecterTextSplitter
from langchain_openAI import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def load_resume():
    loader=PyPDFLoader("resume.pdf")

    documents = loader.load()

    return documents

def doc_splitter(documents):

    splitter = RecursiveCharecterTextSplitter(chunk_size =1000, chunk_overlap=100)

    chunks=splitter.split_documents(documents)

    return chunks

def embedding():
    embedding=OpenAIEmbeddings(model ="text-embedding-3-small")

    return embedding

def vector_db(chunks,embedding):
    vec_db=FAISS.from_documents(documents = chunks, embedding=embedding)

    return vec_db

def load_llm():

    llm=ChatOpenAI(model="gpt-40-mini")

    return llm

prompt= ChatPromptTemplate.from_template(""" you are an AI assistant answering que based on the document provided
use that provided context only. DO NOT MAKE UP ANY INFORMATION FROM YOUR KNOWLEDGE. If the ans is not available in the document context, say like 'it is not mentioned in the document
context:{context}

question : {question}
Answer:""")

def ans(question,llm, vec_db):

    docs=vec_db.similarity_search(question, k=5)

    if not docs:
        return " NO relavant infp found"

    context = "\n\n".join(doc.page_content for doc in docs)


    msg= prompt.format_message(context=context, question= question)

    response = llm.invoke(msg)

    return response.content

def main():
    document = load_resume()

    chunks= doc_splitter

    embedding = embedding()

    vect__DB=vector_db(chunks, embedding)

    llm=load_llm()

    while True():
        question = input("\n You:   ")

        if question.lower() in ["exit", "quit"]:
            break

        ans=ans(question,llm, vect__DB)

        print("Ans:", ans)

if __name__=="__main__":
    main()