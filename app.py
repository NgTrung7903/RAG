import os
import time
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configure the web page
st.set_page_config(page_title="LegalGPT", page_icon="⚖️", layout="wide")

st.title("⚖️ LegalGPT - Trợ lý Luật sư AI")
st.markdown("Hãy tải lên một tài liệu pháp lý (PDF) và hỏi tôi bất kỳ câu hỏi nào về nó!")

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
        
    uploaded_file = st.file_uploader("Tải lên file PDF của bạn", type=["pdf"])
    
    if st.button("Xử lý Dữ liệu", use_container_width=True):
        if not api_key:
            st.error("Lỗi: Thiếu API Key.")
        elif not uploaded_file:
            st.error("Vui lòng tải lên một file PDF.")
        else:
            with st.spinner("Đang xử lý tài liệu..."):
                # Lưu file PDF người dùng tải lên vào một file tạm thời
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name

                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
                
                st.info("⏳ Đang đọc và cắt nhỏ tài liệu PDF...")
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
                chunks = text_splitter.split_documents(docs)
                
                st.info(f"🧠 Đang mã hóa {len(chunks)} đoạn văn bản...")
                vector_store = None
                batch_size = 5
                
                # UI Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    current_batch = i//batch_size + 1
                    total_batches = (len(chunks) + batch_size - 1)//batch_size
                    
                    status_text.text(f"Đang nhúng lô {current_batch}/{total_batches}...")
                    progress_bar.progress(current_batch / total_batches)
                    
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
                                status_text.warning("Phát hiện quá tải API. Tự động nghỉ 60 giây...")
                                time.sleep(60)
                            else:
                                st.error(f"Lỗi: {e}")
                                break
                    
                    # Tránh quá tải
                    if i + batch_size < len(chunks):
                        time.sleep(5)
                        
                if vector_store:
                    # Lưu vào RAM (session_state) thay vì lưu ra ổ cứng để người dùng khác không bị trùng
                    st.session_state['vector_store'] = vector_store
                    st.success("✅ Đã xử lý xong tài liệu! Bạn có thể bắt đầu hỏi.")
                
                # Xóa file tạm sau khi xử lý xong
                os.remove(temp_file_path)

# --- MAIN CHAT INTERFACE ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt_text := st.chat_input("Hỏi câu hỏi pháp lý..."):
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
            with st.spinner("Đang phân tích hồ sơ pháp lý..."):
                vector_store = st.session_state['vector_store']
                retriever = vector_store.as_retriever(search_kwargs={"k": 40})
                
                llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.1)
                
                system_prompt = (
                    "Bạn là LegalGPT - một Chuyên gia Pháp lý và Luật sư Tư vấn cấp cao.\n"
                    "Nhiệm vụ của bạn là tư vấn, giải đáp thắc mắc pháp lý TRÊN CƠ SỞ NGỮ CẢNH ĐƯỢC CUNG CẤP.\n\n"
                    "NGUYÊN TẮC HOẠT ĐỘNG:\n"
                    "1. CHÍNH XÁC & KHÁCH QUAN: Tuyệt đối không tự bịa ra điều luật, mức phạt hay quy định nếu không có trong ngữ cảnh.\n"
                    "2. TRÍCH DẪN CƠ SỞ: Luôn cố gắng trích dẫn cụ thể (Điều mấy, Khoản mấy, Chương nào) nếu ngữ cảnh có đề cập.\n"
                    "3. RÕ RÀNG & DỄ HIỂU: Trình bày mạch lạc bằng gạch đầu dòng, giải thích các thuật ngữ pháp lý phức tạp cho người dân bình thường hiểu.\n"
                    "4. BẢO LƯU: Nếu thông tin trong ngữ cảnh không đủ để trả lời chính xác, hãy nói rõ: 'Dựa trên tài liệu được cung cấp, không có đủ thông tin quy định về vấn đề này...'\n\n"
                    "NGỮ CẢNH PHÁP LÝ:\n{context}"
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
