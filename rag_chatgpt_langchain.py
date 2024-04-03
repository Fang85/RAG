import os
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

openai.api_key = os.environ['OPENAI_API_KEY']

llm_model = "gpt-3.5-turbo-1106"
llm = ChatOpenAI(temperature=0.0, model=llm_model)


file = "Exploring the Boundaries of GPT-4 in Radiology.pdf"

loader = PyPDFLoader()
all_doc = loader.load_and_split(file)

vectorstore = Chroma.from_documents(documents=all_doc, embedding=GPT4AllEmbeddings())
print("data are stored in database!")

prompt = PromptTemplate.from_template(
    """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
)

def format_docs(docs):
    format_content = "\n\n".join(doc.page_content for doc in docs)
    reference_page = "Page "+", ".join(str(doc.metadata['page']) for doc in docs)
    return format_content,reference_page

chain =  prompt | llm | StrOutputParser()

question ="What is the disadvantage of using chatgpt 4 in radilology?"

docs = vectorstore.similarity_search(question,k=2)# filter ={}
format_content, reference_page = format_docs(docs)

# Run
response=chain.invoke({"context": format_content, 'question': question})
print(f"Answer: {response}\nReference: {reference_page}.\n")
