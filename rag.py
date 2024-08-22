#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install langchain langchain-community langchain-core sentence_transformers langchain-openai python-dotenv beautifulsoup4 chromadb


# In[2]:



# In[3]:


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings


# from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # change model and embedding #c1
from langchain_community.embeddings import OllamaEmbeddings

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter

from urllib.parse import urlparse, unquote

import os
from dotenv import load_dotenv
load_dotenv()


# model_local = ChatOllama(model="mistral") #c1
model_local = ChatOpenAI(   
    model_name="gpt-4o-mini",
    temperature = 1
    )

# 1. Split data into chunks
urls = [
    "https://frackinuniverse.miraheze.org/wiki/Main_Page",
    "https://frackinuniverse.miraheze.org/wiki/Getting_Started", 
    "https://frackinuniverse.miraheze.org/wiki/Personal_Tricorder",
    "https://frackinuniverse.miraheze.org/wiki/The_Player",
    "https://frackinuniverse.miraheze.org/wiki/Stars",
    "https://frackinuniverse.miraheze.org/wiki/Crafting",
    "https://frackinuniverse.miraheze.org/wiki/Combat",
    "https://frackinuniverse.miraheze.org/wiki/Weapons"
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# extract page name
for item in docs_list:
    full_url = item.metadata.get("source")
    parsed_url = urlparse(full_url)
    page_name = unquote(parsed_url.path.split('/')[-1])

    item.metadata["id"] = page_name
    print(item.metadata["id"])

# TODO: experiment with chunk size

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
doc_splits = text_splitter.split_documents(docs_list)



# In[7]:


# 2. Convert documents to Embeddings and store them
# using openAI embeddings
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-openai",
    embedding= OpenAIEmbeddings(),
)

# using bge as embeddings
# model_name = "BAAI/bge-large-en-v1.5"
# # model_kwargs = {'device': 'cuda'} # HuggingFace Transformers uses CPU by default
# encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
# bge_embeddings = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     encode_kwargs=encode_kwargs
#     # query_instruction="为这个句子生成表示以用于检索相关文章："
# )
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-bge-embed",
#     embedding= bge_embeddings,
# )

retriever = vectorstore.as_retriever()

def retrieve_and_format(query):
    relevant_docs = retriever.invoke(query)
    print('\n\n**********SOURCES**********')
    print([doc.metadata.get("id") for doc in relevant_docs])
    return "\n\n".join([doc.page_content for doc in relevant_docs])

# question = "List out all one-handed melee weapons"
question = "what are the features of the personal tricorder?"
exit_keyword = "exit"

# 4. After RAG
rag_template = """Answer the question based only on the following context. If the information is not in the context, say you don't have that information.
:
{context}
Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = (
    {"context": RunnableLambda(retrieve_and_format), "question": RunnablePassthrough()}
    | rag_prompt
    | model_local
    | StrOutputParser()
)
print(rag_chain.invoke(question))

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


# #### UI

# In[ ]:


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

