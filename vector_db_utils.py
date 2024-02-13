from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from custom_embedder import CustomEmbedder

 # create the open-source embedding function
#embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

embedding_function = CustomEmbedder()

def load_docuement_to_db(doc_path):
    # load the document and split it into chunks
    #loader = TextLoader("resources/state_of_the_union.txt")
    loader = TextLoader(doc_path)
    documents = loader.load()
    # split it into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    embed_documents = text_splitter.split_documents(documents)
    # load it into Chroma
    db = Chroma.from_documents(documents=embed_documents, embedding=embedding_function, persist_directory="./chromadb")
    #db = Chroma.from_documents(embed_documents, embedding_function, persist_directory="./chroma_db")


def query_vector_db(query):
    db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    # query it
    #query = "What did the president say about Ketanji Brown Jackson"
    docs = db3.similarity_search(query,)

    # print results
    print(docs)
    return docs[0].page_content
def as_retriever():
    db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    return db3.as_retriever(search_kwargs={"k": 1})