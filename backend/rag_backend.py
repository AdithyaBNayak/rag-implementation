import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorstoreIndexCreator, VectorStoreIndexWrapper
from langchain.llms.bedrock import Bedrock


def get_document_index():
    # Construct the path to the PDF file dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_file_path = os.path.join(current_dir, '..', 'pdf_datas', 'IIFL Wealth Hurun India Rich List 2022- Media Release.pdf')

    # Loading the pdf data in PyPDFLoader Object
    pdf_loader_data = PyPDFLoader(pdf_file_path)

    # Creating Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", "" ],
        chunk_size = 100,
        chunk_overlap = 10
    )

    # Instantiating BedrockEmbedding
    bedrock_embedding = BedrockEmbeddings(
        credentials_profile_name="default",
        model_id="amazon.titan-embed-text-v1"
    )

    # Splitting the Db, Creating VectorDB, Creting Embeddings and Storing it using VectorStoreIndexCreator
    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        text_splitter=text_splitter,
        embedding=bedrock_embedding
    )

    db_index = index_creator.from_loaders([pdf_loader_data])
    return db_index

def fm_llm():
    # Create an LLM for Bedrock Model
    return Bedrock(
        credentials_profile_name='default',
        model_id='meta.llama2-13b-chat-v1',
        model_kwargs={
            # "max_tokens_to_sample": 300,
            # "temperature": 0.1,
            # "top_p": 0.9,
            "max_gen_len": 512,
            "temperature": 0.1,
            "top_p": 0.9
        })
    
def doc_rag_response(index: VectorStoreIndexWrapper, prompt):
    # Query from the index
    hr_rag_query  = index.query(
        question=prompt,
        llm=fm_llm()
    )
    return hr_rag_query