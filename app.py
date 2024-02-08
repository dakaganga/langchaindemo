from customllm import CustomLLM

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import vector_db_utils
'''
question = "Who won the FIFA World Cup in the year 1994? "

template = """Question: {question}

"""

'''

question="What did the president say about Ketanji Brown Jackson"
vector_db_utils.load_docuement_to_db("resources/state_of_the_union.txt")
search_results=vector_db_utils.query_vector_db(question)

template = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: {search_results}

Question: {question}

Answer: """

prompt = PromptTemplate(template=template,input_variables=["question"])

llm = CustomLLM()
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.invoke(search_results[0],question))