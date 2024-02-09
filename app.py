from customllm import CustomLLM

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import vector_db_utils
'''
question = "Who won the FIFA World Cup in the year 1994? "

template = """Question: {question}

"""

'''

question="what Zelenskyy said to EU pariliment"
vector_db_utils.load_docuement_to_db("resources/state_of_the_union.txt")
search_results=vector_db_utils.query_vector_db(question)

template = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: {context}

Question: {question}

Answer: """

prompt = PromptTemplate(template=template,input_variables=["context","question"])

llm = CustomLLM()

llm_chain = LLMChain(prompt=prompt, llm=llm)
response=llm_chain.invoke({"context":search_results,"question":question})
print(f"Question:{response['question']} {response['text']}")