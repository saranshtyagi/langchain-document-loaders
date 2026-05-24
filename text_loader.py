from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

model = ChatHuggingFace(llm = llm)

loader = TextLoader('poem.txt', encoding='utf-8')

docs = loader.load()

prompt = PromptTemplate(
    template="Write a summary of the {poem}", 
    input_variables=["poem"]
)

parser = StrOutputParser()

print(type(docs))
print(len(docs))
print(docs[0].metadata)
print(docs[0].page_content)

chain = prompt | model | parser

result = chain.invoke({'poem': docs[0].page_content})

print(result)       