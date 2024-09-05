#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install -U -q langchain langchain-community langchain-core sentence_transformers langchain-openai python-dotenv beautifulsoup4 langchain-chroma langchain-groq langchain-ollama')


# In[2]:


# https://mer.vin/2024/02/ollama-embedding/ - original source code
import sys
print(f"Python interpreter: {sys.executable}") # getting python interpreter

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community import embeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings


# from langchain_community.chat_models import ChatOllama
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

import os
from dotenv import load_dotenv
from pathlib import Path  


# In[3]:


load_dotenv()

# current_dir = os.path.dirname(os.path.abspath(__file__))
# db_dir = os.path.join(current_dir, "db")
# persistent_directory = os.path.join(db_dir, "chromadb-with-sources")
persistent_directory = "C:\\Users\\User-PC\\Documents\\Overseas stuff\\Edinburgh\\Year 3\\coding projects\\Ollama RAG\\db\\chromadb"


def extract_page_name(docs):
    for item in docs: 
        full_url = item.metadata.get("source")
        parsed_url = urlparse(full_url)
        page_name = unquote(parsed_url.path.split('/')[-1])

        item.metadata["id"] = page_name
        print(item.metadata["id"])
    
    #for Jina AI
    # for item in docs: # using WebBaseLoader
    #     full_url = item.metadata.get("source")
    #     parsed_url = urlparse(full_url)
        # page_name = unquote(.split('/')[-1])

    #     item.metadata["id"] = page_name
    #     print(item.metadata["id"])
    


# In[4]:


# TEST
# import requests

# def scrapin(url: str) -> str:                    
#   response = requests.get("https://r.jina.ai/" + url)
# #   return response.text
#   return response.url

# scraped_biome=scrapin("https://frackinuniverse.miraheze.org/wiki/Biomes")
# print(scraped_biome)


# In[27]:


# embedding = OpenAIEmbeddings( model = "text-embedding-3-small") # using openAI embedding

model_name = "BAAI/bge-large-en-v1.5"           # using bge as embedding
encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

import requests
from langchain.docstore.document import Document  

# from LLM for devs: https://github.com/trancethehuman/ai-workshop-code/blob/main/Web_scraping_for_LLM_in_2024.ipynb


# claude's suggestion
def scrape_jina_ai(url: str) -> Document:  
    response = requests.get("https://r.jina.ai/" + url)  
    return Document(page_content=response.text, metadata={"source": url}) 

# if not os.path.exists(persistent_directory):    # setting up vector store
if True:    # setting up vector store
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
        "https://frackinuniverse.miraheze.org/wiki/Planets",
        "https://frackinuniverse.miraheze.org/wiki/Biomes"
    ]
    # loading urls 
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    # docs_list = [scrape_jina_ai(url) for url in urls] # scrape using Jina AI's reader
    print("BIOMES PAGE >>> ")
    print(docs_list[-1].page_content+"\n")

    # set page name in metadata
    extract_page_name(docs_list)

    # split document into chunks
    # TODO: experiment with chunk size
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(docs_list)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(doc_splits)}")
    print(f"Sample chunk\n{doc_splits[0]}\n")


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
    




# In[48]:


vectorstore = Chroma(persist_directory=persistent_directory,
            embedding_function=embedding)
retriever = vectorstore.as_retriever(k=10)  # initialize retriever

def retrieve_and_format(query):
    relevant_docs = retriever.invoke(query)
    print('\n>>>> SOURCES <<<<< :')
    print([doc.metadata.get("id") for doc in relevant_docs])
    print_page_contents(relevant_docs)
    return "\n\n".join([doc.page_content for doc in relevant_docs])

def print_page_contents(docs):
    i=1
    for doc in docs:
        print(f"======= Doc {i} =======")
        print(doc.page_content)
        # print(doc.page_content[:100])
        i+=1
        


# model_local = ChatOpenAI(   
#     model="gpt-4o-mini",
#     temperature = 0.1
#     )

model_local = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-8b-8192", # this > llama 3.1 8b-instant (performance)
        # model="llama-3.1-8b-instant",
        # model="llama-3.1-70b-versatile",
        temperature=0.1,
        )

# model_local = ChatOllama(
#                 model="llama3.1",
#                 temperature=0.5)



# In[49]:


# question = "List out all one-handed melee weapons"
question = "what are the features of the personal tricorder?"
exit_keyword = "exit"

# 4. After RAG
rag_template = """
                Role: You're the best Starbound Frackin' Universe enthusiast who's really pedantic and likes details but with concise and to-the-point sentences.
                Task:  
                Focus on the MAIN contents of the body, ignoring the periphery i.e., navigation bars, headers, footers, social media etc. 
                Give me the answer straight-away without any conversational preamble.
                Answer the question based only on the following context. If the information is not in the context, it is CRITICAL that you say you don't have the information
                :
                {context}
                Question: {question}
                """
                
llama_template = """
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a pedantic but knowledgeable, efficient and direct AI assistant for Frackin' Universe Website. Provide concise answers focusing on key information. Offer tactful suggestions to solve the user's question. 
                Answer the *question* based only on the following *context*. If the information is not in the *context*, say that you don't have the informmation.
                Focus on the MAIN contents of the given webpage in the *context*, ignoring the periphery of the website i.e., navigation bars, headers, footers, social media etc. 
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

# while True:  # to have an ongoing conversation
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


# In[50]:


tricorder_2 = "where can i find this tricorder if i happen to lose it?"
result_2 = rag_chain.invoke(tricorder_2)
print(result_2)


# In[51]:


tricorder_3 = "Can i craft a personal tricorder without having to use my pixels?"
result_3 = rag_chain.invoke(tricorder_3)
print(result_3)


# In[52]:


tricorder_4 = "can you tell me a few things that i can craft (along with its  required crafting amterials) with the tricorder?"
result_4 = rag_chain.invoke(tricorder_4)
print(result_4)


# In[53]:


# Stars
q5 = "what kinds of planets are in gentle stars?"
r5 = rag_chain.invoke(q5)
print(r5)


# In[54]:


# Planets
q6 = "what locations can i find gelatinous planets?" # only takes sources from the Stars page (in Planets, it says Gentle Stars, Temperate Stars etc.)
r6 = rag_chain.invoke(q6) 
print(r6)


# In[55]:


# Planets
# q7 = "what's the fauna threat for gelatinous planets?"
q7 = "what's the highest tier for a normal volcaninc planet?"
r7 = rag_chain.invoke(q7)
print(r7)


# In[56]:


# Biomes
q8 = "what's the reason to visit for Bog biomes?"
r8 = rag_chain.invoke(q8)
print(r8)


# #### UI

# In[57]:


import gradio as gr
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
# from langchain_community.chat_models import ChatOllama
from langchain.llms import OpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter

def process_input(urls, question):
    model_local = ChatOllama(model="mistral")
    
    # Convert string of URLs to list
    urls_list = urls.split("\n")
    docs = [WebBaseLoader(url).load() for url in urls_list]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()

    after_rag_template = """Answer the question based only on the following context (If the information is not in the context, say you don't have that information):
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

# Define Gradio interface
iface = gr.Interface(fn=process_input,
                     inputs=[gr.Textbox(label="Enter URLs separated by new lines"), gr.Textbox(label="Question")],
                     outputs="text",
                     title="Document Query with Ollama",
                     description="Enter URLs and a question to query the documents.")
iface.launch()

