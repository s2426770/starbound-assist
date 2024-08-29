#!/usr/bin/env python
# coding: utf-8

# In[28]:


# get_ipython().run_line_magic('pip', 'install -U langchain langchain-community langchain-core sentence_transformers langchain-openai python-dotenv beautifulsoup4 langchain-chroma langchain-groq langchain-ollama')


# In[29]:


# https://mer.vin/2024/02/ollama-embedding/ - original source code
import sys
print(f"Python interpreter: {sys.executable}") # getting python interpreter

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community import embeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings


from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # change model and embedding #c1
from langchain_community.embeddings import OllamaEmbeddings

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from urllib.parse import urlparse, unquote

import requests

import os
from dotenv import load_dotenv
from pathlib import Path  


# In[30]:


load_dotenv()

persistent_directory = "C:\\Users\\User-PC\\Documents\\Overseas stuff\\Edinburgh\\Year 3\\coding projects\\Ollama RAG\\db\\chromadb"


def extract_page_name(docs):
    for item in docs:
        full_url = item.metadata.get("source")
        parsed_url = urlparse(full_url)
        page_name = unquote(parsed_url.path.split('/')[-1])

        item.metadata["id"] = page_name
        print(item.metadata["id"])


# In[31]:


# using openAI embedding
embedding = OpenAIEmbeddings( model = "text-embedding-3-small")

# using bge as embeddings
# model_name = "BAAI/bge-large-en-v1.5"
# encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
# embedding = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     encode_kwargs=encode_kwargs
# )

def scrape_jina_ai(url: str) -> str:
  response = requests.get("https://r.jina.ai/" + url)
  return response.text

# setting up vector store
# if True:
if not os.path.exists(persistent_directory):
    print("Persistent directory doesn't exist. Initializing vector store...")

    urls = [
        "https://frackinuniverse.miraheze.org/wiki/Main_Page",
        "https://frackinuniverse.miraheze.org/wiki/Getting_Started", 
        "https://frackinuniverse.miraheze.org/wiki/Personal_Tricorder",
        "https://frackinuniverse.miraheze.org/wiki/The_Player",
        "https://frackinuniverse.miraheze.org/wiki/Stars",
        "https://frackinuniverse.miraheze.org/wiki/Crafting",
        "https://frackinuniverse.miraheze.org/wiki/Combat",
        "https://frackinuniverse.miraheze.org/wiki/Weapons",
        "https://frackinuniverse.miraheze.org/wiki/Planets"
    ]
    # loading urls 
    docs = [WebBaseLoader("https://r.jina.ai/" + url).load() for url in urls] # scrape using langchain
    # docs = [scrape_jina_ai(url) for url in urls] # scrape using Jina AI
    docs_list = [item for sublist in docs for item in sublist]

    # set page name in metadata
    extract_page_name(docs_list)
    # for item in docs_list:
    #     full_url = item.metadata.get("source")
    #     parsed_url = urlparse(full_url)
    #     page_name = unquote(parsed_url.path.split('/')[-1])

    #     item.metadata["id"] = page_name
    #     print(item.metadata["id"])
    

    # split document into chunks
    # TODO: experiment with chunk size
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(doc_splits)}")
    print(f"Sample chunk\n{doc_splits[0].page_content}\n")



    # Convert documents to Embeddings and store them
    print("\n--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        # collection_name="rag-openai",
        persist_directory=persistent_directory,
        embedding = embedding,
    )
    print("\n--- Finished creating vector store ---")
else: 
    print("Vector store already exists. No need to initialize")



# In[44]:


vectorstore = Chroma(persist_directory=persistent_directory,
            embedding_function=embedding)
retriever = vectorstore.as_retriever()  # initialize retriever

def retrieve_and_format(query):
    relevant_docs = retriever.invoke(query)
    print('\n\n**********SOURCES**********')
    print([doc.metadata.get("id") for doc in relevant_docs])
    # print_docs(relevant_docs)
    return "\n\n".join([doc.page_content for doc in relevant_docs])

def print_docs(docs):
    i=1
    for doc in docs:
        print(f"---> Doc {i} (Source: {doc.metadata.get("id")}) :\n")
        print(doc.page_content[:100])
        i+=1
        
# model_local = ChatOllama(model="mistral") #c1
# model_local = ChatOpenAI(   
#     model="gpt-4o-mini",
#     temperature = 0.5
#     )

# llm = ChatOpenAI()

model_local = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        # model="llama-3.1-70b-versatile",o-0
        temperature=0,
        )

# model_local = ChatOllama(
#                 model="llama3.1",
#                 temperature=0.5)



# In[45]:


# question = "List out all one-handed melee weapons"
question = "what are the features of the personal tricorder?"
exit_keyword = "exit"

# 4. After RAG
rag_template = """
                Role: You're the best Starbound Frackin' Universe enthusiast who's really pedantic and likes details but with concise and to-the-point sentences.
                Task:  
                Answer the question based only on the following context. If the information is not in the context, it is CRITICAL that you say you don't have the information. Focus on the MAIN contents of the body, ignoring the periphery i.e., navigation bars, headers, footers, social media etc. 
                Give me the answer straight-away without any conversational preamble.
                Question: {question}
                Context:
                {context}
                """
                
llama_template = """
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a pedantic but knowledgeable, efficient and direct AI assistant for Frackin' Universe Website. Provide concise answers focusing on key information. Offer tactful suggestions to solve the user's question. 
                Answer the *question* based only on the following *context*. If the information is not in the *context*, say that you don't have the informmation.
                Focus on the MAIN contents of the given *context*, ignoring the periphery of the website i.e., navigation bars, headers, footers, social media etc. 
                It is **CRITICAL** that you thoroughly digest the given *context* to answer the user's *question*.
                Context: {context}
                <|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                {question}
                <|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>
                """
                
# rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_prompt = ChatPromptTemplate.from_template(llama_template)
rag_chain = (
    {"context": RunnableLambda(retrieve_and_format), "question": RunnablePassthrough()}
    | rag_prompt
    | model_local
    | StrOutputParser()
)
result = rag_chain.invoke(question)

print("********ANSWER********")
print(result)

# while True:
    # user_question = input("\nHuman: ").strip()

    # if user_question.lower() == exit_keyword.lower():
    #     print("Exiting the program. Goodbye!")
    #     break

    # if user_question:
    #     print("\nProcessing your question...\n")
    #     nomic_response = rag_chain.invoke(user_question)
    #     openai_response = rag_chain_openai.invoke(user_question)
    #     print("====== Nomic Answer======:\n ",nomic_response)
    #     print("\n====== OpenAI Answer ======\n",openai_response)
    # else:
    #     print("Please enter a valid question")

# loader = PyPDFLoader("Ollama.pdf")
# doc_splits = loader.load_and_split()


# In[46]:


tricorder_2 = "where can i find this tricorder if i happen to lose it?"
result_2 = rag_chain.invoke(tricorder_2)
print(result_2)


# In[47]:


tricorder_3 = "Can i craft a personal tricorder without having to use my pixels?"
result_3 = rag_chain.invoke(tricorder_3)
print(result_3)


# In[48]:


tricorder_4 = "can you tell me a few things that i can craft (along with its  required crafting amterials) with the tricorder?"
result_4 = rag_chain.invoke(tricorder_4)
print(result_4)


# #### UI

# In[49]:


# import gradio as gr
# from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
# from langchain_community.vectorstores import Chroma
# from langchain_community import embeddings
# # from langchain_community.chat_models import ChatOllama
# from langchain.llms import OpenAI
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.output_parsers import PydanticOutputParser
# from langchain.text_splitter import CharacterTextSplitter

# def process_input(urls, question):
#     model_local = ChatOllama(model="mistral")
    
#     # Convert string of URLs to list
#     urls_list = urls.split("\n")
#     docs = [WebBaseLoader(url).load() for url in urls_list]
#     docs_list = [item for sublist in docs for item in sublist]
    
#     text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
#     doc_splits = text_splitter.split_documents(docs_list)

#     vectorstore = Chroma.from_documents(
#         documents=doc_splits,
#         collection_name="rag-chroma",
#         embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
#     )
#     retriever = vectorstore.as_retriever()

#     after_rag_template = """Answer the question based only on the following context (If the information is not in the context, say you don't have that information):
#     {context}
#     Question: {question}
#     """
#     after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
#     after_rag_chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | after_rag_prompt
#         | model_local
#         | StrOutputParser()
#     )
#     return after_rag_chain.invoke(question)

# # Define Gradio interface
# iface = gr.Interface(fn=process_input,
#                      inputs=[gr.Textbox(label="Enter URLs separated by new lines"), gr.Textbox(label="Question")],
#                      outputs="text",
#                      title="Document Query with Ollama",
#                      description="Enter URLs and a question to query the documents.")
# iface.launch()

