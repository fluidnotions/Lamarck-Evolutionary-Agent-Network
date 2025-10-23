from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    base_url="https://822jvq5wcyw2r3-8000.proxy.runpod.net/v1",
    api_key="none",          # or your key if needed
    temperature=0.6
)

response = llm.invoke("Explain Lamarckian inheritance for AI agents.")
print(response.content)
