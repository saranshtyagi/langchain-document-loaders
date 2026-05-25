import os
os.environ["USER_AGENT"] = "Mozilla/5.0"
from langchain_community.document_loaders import WebBaseLoader
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

prompt = PromptTemplate(
    template='Answer the following {question} from the {text}', 
    input_variables=['question', 'text']
)

parser = StrOutputParser()

url='https://example.com/'
loader = WebBaseLoader(url)

docs = loader.load()

# print(docs[0].page_content)

chain = prompt | model | parser

result = chain.invoke({'question': 'What is the website intended for?', 'text': docs[0].page_content})

print(result)