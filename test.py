from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings

# from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # change model and embedding #c1
from langchain_community.embeddings import OllamaEmbeddings

from langchain_core.runnables import RunnablePassthrough
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
    temperature = 0.8
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
    "https://frackinuniverse.miraheze.org/wiki/Weapons",
    "https://starbounder.org/Starbound_Wiki",
    "https://starbounder.org/Combat"
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
print(type(docs_list[0]))

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
doc_splits = text_splitter.split_documents(docs_list)

# extract page name
for item in docs_list:
    full_url = item.metadata.get("source")
    parsed_url = urlparse(full_url)
    page_name = unquote(parsed_url.path.split('/')[-1])

    item.metadata["id"] = page_name
    print(item.metadata["id"])


# 2. Convert documents to Embeddings and store them
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding= OllamaEmbeddings(model='nomic-embed-text'),
)

# using openAI embeddings
vectorstore_openai = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-openai",
    embedding= OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()
retriever_openai = vectorstore_openai.as_retriever()

# question = "Can you detail more on the newly-added races in that FU introduces?"

# 4. After RAG
# print("\n########\nembed: nomic embed text\n")
rag_template = """Answer the question based only on the following context. If the information is not in the context, say you don't have that information.
:
{context}
Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | model_local
    | StrOutputParser()
)
# print(rag_chain.invoke(question))

# print("\n########\nembed: open ai\n")
rag_chain_openai = (
    {"context": retriever_openai, "question": RunnablePassthrough()}
    | rag_prompt
    | model_local
    | StrOutputParser()
)
# print(rag_chain_openai.invoke(question))

exit_keyword = "exit"
while True:
    print(f"Enter your questions Type '{exit_keyword}' to end the program.")
    user_question = input("\nHuman: ").strip()

    if user_question.lower() == exit_keyword.lower():
        print("XXXXXXXXXXX Exiting the program. Goodbye! XXXXXXXXXX")
        break

    if user_question:
        print("\nProcessing your question...\n")

        nomic_response = rag_chain.invoke(user_question)
        openai_response = rag_chain_openai.invoke(user_question)

        print("====== Nomic Answer======\n ",nomic_response)
        print("\n====== OpenAI Answer ======\n",openai_response)
    else:
        print("Please enter a valid question")

# loader = PyPDFLoader("Ollama.pdf")
# doc_splits = loader.load_and_split()

