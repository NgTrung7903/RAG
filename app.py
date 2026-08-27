import os
import time
import tempfile
import gc
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configure the web page
st.set_page_config(page_title="DocumentAI", page_icon="📄", layout="wide")

st.title("📄 DocumentAI - Trợ lý Đọc hiểu Tài liệu")
st.markdown("Hãy tải lên bất kỳ tài liệu nào (PDF) và hỏi tôi mọi thứ về nội dung bên trong!")

# Lấy API Key từ Streamlit Secrets (Bảo mật, không lộ trên Github)
api_key = None
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
elif "GOOGLE_API_KEY" in os.environ:
    api_key = os.environ["GOOGLE_API_KEY"]

# Đặt vào biến môi trường để Langchain có thể sử dụng
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Tải Tài Liệu")
    
    if not api_key:
        st.error("⚠️ Hệ thống chưa được cấu hình API Key. Vui lòng liên hệ quản trị viên!")
        
    uploaded_files = st.file_uploader("Tải lên các file PDF của bạn", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Xử lý Dữ liệu", use_container_width=True):
        if not api_key:
            st.error("Lỗi: Thiếu API Key.")
        elif not uploaded_files:
            st.error("Vui lòng tải lên ít nhất một file PDF.")
        else:
            with st.spinner("Đang xử lý tài liệu..."):
                st.info("⏳ Đang tải mô hình nhúng (Embedding Model) Local...")
                embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
                
                vector_store = None
                batch_size = 32
                status_text = st.empty()
                total_chunks_processed = 0
                
                for file_obj in uploaded_files:
                    st.info(f"📄 Đang xử lý file: {file_obj.name}...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(file_obj.getvalue())
                        temp_file_path = temp_file.name
                        
                    loader = PyMuPDFLoader(temp_file_path)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
                    
                    for doc_page in loader.lazy_load():
                        page_chunks = text_splitter.split_documents([doc_page])
                        
                        for i in range(0, len(page_chunks), batch_size):
                            batch = page_chunks[i:i+batch_size]
                            total_chunks_processed += len(batch)
                            
                            status_text.text(f"Đang nhúng... (Đã băm và xử lý {total_chunks_processed} đoạn văn bản từ các file)")
                            
                            if vector_store is None:
                                vector_store = FAISS.from_documents(batch, embeddings)
                            else:
                                vector_store.add_documents(batch)
                                
                        if vector_store:
                            vector_store.save_local("faiss_index_temp")
                            
                        del page_chunks
                        gc.collect()
                        
                    os.remove(temp_file_path)
                        
                if vector_store:
                    st.session_state['vector_store'] = vector_store
                    st.success("✅ Đã xử lý xong toàn bộ tài liệu! Bạn có thể bắt đầu hỏi.")

# --- MAIN CHAT INTERFACE ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt_text := st.chat_input("Hỏi câu hỏi về tài liệu..."):
    if not api_key:
        st.error("⚠️ Hệ thống đang bảo trì (Thiếu API Key).")
    elif 'vector_store' not in st.session_state:
        st.warning("⚠️ Vui lòng tải file PDF lên và ấn 'Xử lý' ở thanh bên trái trước.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Đang phân tích tài liệu..."):
                vector_store = st.session_state['vector_store']
                retriever = vector_store.as_retriever(search_kwargs={"k": 40})
                
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
                
                chain_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, chain_prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                # Get the answer
                response = rag_chain.invoke({"input": prompt_text})
                answer = response.get('answer', '')
                
                # Display and save answer
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
