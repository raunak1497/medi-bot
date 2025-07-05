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

