import os
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')
import gc
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"

embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")

if os.path.exists("faiss_index"):
    print("✅ Vector data found. Loading from disk (no API cost)...")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    print("⏳ STEP 1 & 2 & 3: Loading and embedding PDF document page by page...")
    loader = PyMuPDFLoader("document.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    
    vector_store = None
    batch_size = 32 # Tăng batch_size vì không bị giới hạn API
    total_chunks = 0
    
    for page_idx, doc_page in enumerate(loader.lazy_load()):
        page_chunks = text_splitter.split_documents([doc_page])
        
        for i in range(0, len(page_chunks), batch_size):
            batch = page_chunks[i:i+batch_size]
            total_chunks += len(batch)
            print(f"  -> Embedding up to chunk {total_chunks}...")
            
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
            
        # Save after each page to avoid data loss on crash
        if vector_store:
            vector_store.save_local("faiss_index")
            
        del page_chunks
        gc.collect()
        
    print("✅ Data saved to 'faiss_index'. Next run will skip this step!")

retriever = vector_store.as_retriever(search_kwargs={"k": 15}) 

print("🤖 STEP 4: Initializing LegalGPT brain...")
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.1) 

system_prompt = (
    "You are LegalGPT - a Senior Legal Expert and Consultant Lawyer.\n"
    "Your task is to consult and answer legal questions BASED ON THE PROVIDED CONTEXT.\n\n"
    "OPERATING PRINCIPLES:\n"
    "1. ACCURATE & OBJECTIVE: Absolutely do not fabricate laws, fines, or regulations if they are not in the context.\n"
    "2. CITE BASES: Always try to cite specifics (which Article, Clause, Chapter) if mentioned in the context.\n"
    "3. CLEAR & UNDERSTANDABLE: Present clearly using bullet points, explain complex legal terms for ordinary people to understand.\n"
    "4. RESERVATION: If the information in the context is not enough to answer accurately, clearly state: 'Based on the provided documents, there is not enough information regulating this issue...'\n\n"
    "LEGAL CONTEXT:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("\n" + "="*50)
print("⚖️  LEGAL CONSULTING SYSTEM (LEGAL-GPT) IS READY! ⚖️")
print("="*50)
print("Hint: You can ask about specific cases, fines, or request a summary of a law.")

while True:
    user_question = input("\n🧑‍⚖️ Your question (Type 'exit' to quit): ")
    if user_question.lower() == 'exit':
        print("Goodbye! Have a good day.")
        break
        
    print("⏳ LegalGPT is looking up records and analyzing the law...")
    response = rag_chain.invoke({"input": user_question})
    
    print("\n" + "-"*50)
    print(f"📜 LAWYER'S ANSWER:\n{response.get('answer', '')}")
    print("DEBUG RAW RESPONSE:", repr(response))
    print("-" * 50)