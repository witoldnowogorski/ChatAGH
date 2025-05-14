import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

os.environ["GOOGLE_API_KEY"] = "AIzaSyCv5Pwbk1Jg-UsiMQ6yd76FaJlJzdvDdVs"

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.7)


messages = [
    SystemMessage(content="You are a helpful AI assistant that specializes in explaining complex topics simply."),
    HumanMessage(content="What is machine learning and how does it work?")
]

response = llm.invoke(messages)
print("Chat message response:")
print(response.content)
print("\n" + "-"*50 + "\n")

prompt = ChatPromptTemplate.from_template(
    "You are an expert in {subject}. Explain {concept} in simple terms."
)

chain = LLMChain(llm=llm, prompt=prompt)

response = chain.invoke({
    "subject": "astronomy",
    "concept": "black holes"
})

print("Chain response:")
print(response["text"])
print("\n" + "-"*50 + "\n")

# Streaming responses
print("Streaming response:")
for chunk in llm.stream("Explain the difference between AI, ML, and deep learning"):
    print(chunk.content, end="", flush=True)
print("\n" + "-"*50 + "\n")

# Setting different parameters for generation
llm_creative = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=1.0,     # Higher temperature for more creative responses
    top_p=0.95,          # Nucleus sampling
    top_k=40,            # Top-k sampling
    max_output_tokens=1024  # Maximum length of response
)

response = llm_creative.invoke("Write a short poem about artificial intelligence")
print("Creative response with custom parameters:")
print(response.content)