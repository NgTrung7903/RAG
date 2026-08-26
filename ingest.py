from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("⏳ Reading PDF file...")
loader = PyPDFLoader("document.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200 
)

chunks = text_splitter.split_documents(documents)

print(f"✅ Split document into {len(chunks)} chunks.")
if len(chunks) > 0:
    print(f"First chunk sample: \n{chunks[0].page_content[:200]}...")
else:
    print("⚠️ WARNING: No text found in the PDF. Your file might be a scanned image or lack a text layer.")