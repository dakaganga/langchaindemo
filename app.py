from customllm import CustomLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import vector_db_utils
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

question="what Zelenskyy said to EU pariliment"

#Vector DB Load data and retrieve data
vector_db_utils.load_docuement_to_db("resources/state_of_the_union.txt")
#search_results=vector_db_utils.query_vector_db(question)
retriever = vector_db_utils.as_retriever()

#template
# template = """Answer the question based on the context below. If the
# question cannot be answered using the information provided answer
# with "I don't know".

# Context: {context}

# Question: {question}

# Answer: """

template = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: {context}

Current conversation:
{chat_history}
User: {question}
AI:"""

prompt = PromptTemplate(template=template,input_variables=["context","chat_history","question"])
llm = CustomLLM()

#llm_chain = LLMChain(prompt=prompt, llm=llm)
#response=llm_chain.invoke({"context":search_results,"question":question})
#print(f"Question:{response['question']} {response['text']}")


# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
# print(chain.invoke(question))
#memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

# chain = load_qa_chain(
#     llm, chain_type="stuff", memory=memory, prompt=prompt
# )
# chain({"input_documents": search_results, "question": question}, return_only_outputs=True)

# conversation = ConversationChain(
#     prompt=prompt,
#     llm=llm, 
#     verbose=True, 
#     memory=ConversationBufferMemory(human_prefix="User"),
#     output_parser=StrOutputParser()
# )
#conversation.invoke({"context":retriever,"question":question})

#conversation.predict({"context":retriever,"question":question})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    llm, 
    retriever,
    memory=memory
)

result=qa({"question": question})
print(result['answer'])
result=qa({"question": "who is Zelenskyy"})
print(result['answer'])
print("###############Printing Chat History#########################")
print(result['chat_history'])