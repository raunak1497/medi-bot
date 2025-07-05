from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from nltk.corpus.reader import documents

#Step 1: Load RAW Pdfs

DATA_PATH = "data/"
def load_pdf_files(data_dir):
    loader = DirectoryLoader(data_dir,glob='*.pdf',loader_cls=PyPDFLoader)

    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH);
print("length of PDF pages", len(documents) )

#Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
print("Length of text chunks", len(text_chunks))

#Create vector embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model