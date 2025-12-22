
import os
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1️⃣ Load document
loader = TextLoader("data/sample.txt" ,encoding ="utf-8")
documents = loader.load()

# 2️⃣ Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
docs = text_splitter.split_documents(documents)

# 3️⃣ Embeddings (local, no API)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4️⃣ Vector store
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
 api_version="2024-02-15-preview"
)


# 6️⃣ Prompt
prompt = PromptTemplate.from_template(
    """Answer the question using only the context below.
If the answer is not present, say "Not found in document".

Context:
{context}

Question:
{question}
"""
)

# 7️⃣ RAG Chain (NEW STYLE)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask_question(question: str) -> str:
    return rag_chain.invoke(question)

