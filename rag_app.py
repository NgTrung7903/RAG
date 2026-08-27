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

print("🤖 STEP 4: Initializing DocumentAI brain...")
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.1) 

system_prompt = (
    "Bạn là DocumentAI - một Trợ lý Đọc hiểu và Tóm tắt Tài liệu thông minh.\n"
    "Nhiệm vụ của bạn là giải đáp thắc mắc, trích xuất thông tin TRÊN CƠ SỞ NỘI DUNG TÀI LIỆU ĐƯỢC CUNG CẤP.\n\n"
    "NGUYÊN TẮC HOẠT ĐỘNG:\n"
    "1. CHÍNH XÁC & TRUNG THỰC: Tuyệt đối không tự bịa ra số liệu, sự kiện hay thông tin nếu không có trong tài liệu.\n"
    "2. TRÍCH DẪN RÕ RÀNG: Hãy cố gắng nói rõ thông tin đó nằm ở phần nào, trang nào (nếu tài liệu có đề cập).\n"
    "3. RÕ RÀNG & DỄ HIỂU: Trình bày mạch lạc bằng gạch đầu dòng, giải thích các thuật ngữ chuyên ngành một cách dễ hiểu.\n"
    "4. BẢO LƯU: Nếu thông tin trong tài liệu không đủ để trả lời chính xác, hãy nói rõ: 'Dựa trên tài liệu được cung cấp, không có đủ thông tin về vấn đề này...'\n\n"
    "NỘI DUNG TÀI LIỆU (NGỮ CẢNH):\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("\n" + "="*50)
print("📄  DOCUMENT AI SYSTEM IS READY! 📄")
print("="*50)
print("Hint: You can ask to summarize the document, extract data, or explain concepts.")

while True:
    user_question = input("\n👤 Your question (Type 'exit' to quit): ")
    if user_question.lower() == 'exit':
        print("Goodbye! Have a good day.")
        break
        
    print("⏳ DocumentAI is looking up records and analyzing the document...")
    response = rag_chain.invoke({"input": user_question})
    
    print("\n" + "-"*50)
    print(f"🤖 AI'S ANSWER:\n{response.get('answer', '')}")
    print("DEBUG RAW RESPONSE:", repr(response))
    print("-" * 50)