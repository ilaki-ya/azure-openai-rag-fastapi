from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

response = llm.invoke(
    "Explain what a Large Language Model is in 3 simple bullet points"
)

print(response.content)
