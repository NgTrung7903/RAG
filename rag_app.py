import os
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

if os.path.exists("faiss_index"):
    print("✅ Vector data found. Loading from disk (no API cost)...")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    print("⏳ STEP 1: Loading and splitting PDF document...")
    loader = PyPDFLoader("document.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    chunks = text_splitter.split_documents(docs)

    print(f"🧠 STEP 2 & 3: Encoding {len(chunks)} vector chunks...")
    vector_store = None
    batch_size = 5 
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"  -> Embedding batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} (contains {len(batch)} chunks)...")
        
        success = False
        while not success:
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                success = True
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "Quota" in str(e):
                    print("     [API overload detected (429). System will automatically pause for 60 seconds then retry...]")
                    time.sleep(60)
                else:
                    raise e
        
        if i + batch_size < len(chunks):
            time.sleep(5)
            
    vector_store.save_local("faiss_index")
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