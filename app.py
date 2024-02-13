from customllm import CustomLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import vector_db_utils


question="what Zelenskyy said to EU pariliment"

#Vector DB Load data and retrieve data
vector_db_utils.load_docuement_to_db("resources/state_of_the_union.txt")
#search_results=vector_db_utils.query_vector_db(question)
retriever = vector_db_utils.as_retriever()

#template
template = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: {context}

Question: {question}

Answer: """

prompt = PromptTemplate(template=template,input_variables=["context","question"])
llm = CustomLLM()

#llm_chain = LLMChain(prompt=prompt, llm=llm)
#response=llm_chain.invoke({"context":search_results,"question":question})
#print(f"Question:{response['question']} {response['text']}")


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(chain.invoke(question))