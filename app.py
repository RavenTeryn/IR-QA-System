import os
# --- 1. 配置国内镜像源 (必须放在最前面) ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 2. 页面基础设置 ---
st.set_page_config(
    page_title="智能检索问答系统",
    page_icon="🧠",
    layout="wide"
)

# --- 3. 自定义 CSS (让界面更好看) ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #4B4B4B;
        text-align: center;
        margin-bottom: 20px;
    }
    .source-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        border-left: 5px solid #ff4b4b;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #b8daff;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. 核心逻辑函数 (带缓存) ---
@st.cache_resource
def initialize_system():
    # A. 加载模型 (这里不显示加载文字，而是静默加载，状态在侧边栏显示)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    # B. 读取 data 文件夹下所有 txt 文件
    loader = DirectoryLoader('data/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    if not documents:
        return None, None

    # C. 切分文档 (针对中文优化断句)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,        
        chunk_overlap=50,      
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""] 
    )
    splits = text_splitter.split_documents(documents)
    
    # D. 建立向量索引
    vector_db = FAISS.from_documents(splits, embeddings)
    
    return vector_db, documents

# --- 5. 初始化系统 ---
with st.spinner("系统正在初始化，构建向量索引中..."):
    vector_db, raw_docs = initialize_system()

# --- 6. 侧边栏布局 (系统状态与技术栈) ---
with st.sidebar:
    st.title("⚙️ 系统控制台")
    
    # 技术栈说明 (替换了原来的Loading提示)
    st.markdown("### 🛠️ 技术架构")
    st.info("**Embedding Model:**\n\nBAAI/bge-small-zh-v1.5 (智源中文语义向量)")
    st.info("**Vector Database:**\n\nFAISS (Facebook AI Similarity Search)")
    
    st.markdown("---")
    
    # 知识库状态
    st.markdown("### 📚 知识库状态")
    if raw_docs:
        st.success(f"✅ 已加载文档数: {len(raw_docs)}")
        with st.expander("查看文件列表"):
            for doc in raw_docs:
                file_name = doc.metadata['source'].split('/')[-1] if '/' in doc.metadata['source'] else doc.metadata['source']
                st.text(f"📄 {file_name}")
    else:
        st.error("⚠️ 未检测到文档，请上传 .txt 文件")

# --- 7. 主界面布局 ---
st.markdown('<div class="main-header">🧠 Retrieval-based QA System</div>', unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: grey;'>基于 RAG 架构的维基百科智能问答系统</div>", unsafe_allow_html=True)
st.markdown("---")

# 搜索框区域
col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
with col1:
    query = st.text_input("请输入您的问题：", placeholder="例如：什么是GenAI？")
with col2:
    search_btn = st.button("🔍 开始检索", use_container_width=True)

# --- 8. 检索与结果展示 ---
if (query or search_btn) and vector_db:
    start_time = time.time()
    
    # 核心检索步骤
    # k=4: 获取最相关的4个片段，第1个作为直接答案，后3个作为参考
    results = vector_db.similarity_search(query, k=4)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 显示检索统计
    st.caption(f"🚀 检索完成，耗时 {elapsed_time:.4f} 秒")

    # A. 最佳答案区域 (Top 1 Result)
    st.markdown("### 💡 最佳匹配答案 (Best Answer Passage)")
    
    best_result = results[0]
    best_source = best_result.metadata['source']
    
    # 使用自定义样式的容器
    st.markdown(f"""
    <div class="answer-box">
        <p style="font-size: 1.1em; line-height: 1.6;">{best_result.page_content}</p>
        <hr style="border-top: 1px dashed #bbb;">
        <p style="color: #666; font-size: 0.9em;">📍 <strong>来源文档:</strong> {best_source}</p>
    </div>
    """, unsafe_allow_html=True)

    # B. 更多相关上下文 (Context)
    with st.expander("📖 查看更多相关上下文 (Supporting Context)"):
        for i, doc in enumerate(results[1:], 1):
            source_file = doc.metadata['source']
            st.markdown(f"""
            <div class="source-card">
                <p><strong>相关片段 {i}:</strong> {doc.page_content}</p>
                <p style="font-size: 0.8em; color: grey;">📄 Source: {source_file}</p>
            </div>
            """, unsafe_allow_html=True)

elif not vector_db:
    st.warning("请检查目录下是否存在 .txt 文件。")