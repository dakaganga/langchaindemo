from customllm import CustomLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
import vector_db_utils
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


question="what Zelenskyy said to EU pariliment"
#Vector DB Load data and retrieve data
vector_db_utils.load_docuement_to_db("resources/state_of_the_union.txt")
search_results=vector_db_utils.query_vector_db(question)
#retriever = vector_db_utils.as_retriever()

#template
# template = """Answer the question based on the context below. If the
# question cannot be answered using the information provided answer
# with "I don't know".

# Context: {context}

# Question: {question}

# Answer: """

template = """
You are a Q&A Chat Assistant.
Use the following pieces of context to answer the question at the end.\
Provide answers strictly from given context below. Do not use any external information or source other than the provided context below while giving response to human
If you don't know the answer, just say that you don't know, don't try to make up an answer.\
Don't provide any additional questions and answers in the response. Don't inlcude the prompt in the response

Cotnext: {context}

{chat_history}

Human: {human_input}
AI:"""

prompt = PromptTemplate(input_variables=["chat_history","human_input","context"],template=template)
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
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

chain = load_qa_chain(
     llm, chain_type="stuff", memory=memory, prompt=prompt
 )
docs=[search_results[0]]

result=chain({"input_documents": docs, "human_input": question},return_only_outputs=True)
print("question 1:")
print(result['output_text'])
result=chain({"input_documents": docs, "human_input": "what he said?"},return_only_outputs=True)
print("question 2:")
print(result['output_text'])
result=chain({"input_documents": docs, "human_input": "who is Narendra Modi?"},return_only_outputs=True)
print("question 3:")
print(result['output_text'])

# conversation = ConversationChain(
#     prompt=prompt,
#     llm=llm, 
#     verbose=True, 
#     memory=ConversationBufferMemory(human_prefix="User"),
#     output_parser=StrOutputParser()
# )
#conversation.invoke(question=question, context=search_results)

#conversation.arun(question=question, context=search_results)
#conversation.predict({"context":search_results,"input":question})
#conversation.predict(input=question)
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# qa = ConversationalRetrievalChain.from_llm(
#     llm=llm, 
#     retriever=retriever,
#     chain_type="stuff", 
#     memory=memory
# )
# question="what Zelenskyy said to EU pariliment"
# result=qa({"question": question})
# print("###############Answer 1#########################")
# print(result['answer'])


# result=qa({"question": "what he said?"})
# print("###############Answer 2#########################")
# print(result['answer'])

# result=qa({"question": "Who is Narendra Modi?"})
# print("###############Answer 3#########################")
# print(result['answer'])

# print("###############Printing Chat History#########################")
# print(result['chat_history'])