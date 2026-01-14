import streamlit as st
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. 页面设置 (对应功能要求: User Interface) ---
st.set_page_config(page_title="我的智能问答系统", layout="wide")
st.title("🤖 维基百科智能问答系统 (QA Bot)")
st.write("本系统基于 RAG 技术，能够根据上传的知识库回答问题。")

# --- 2. 加载与处理数据的函数 ---
@st.cache_resource  # 这个装饰器让系统不用每次刷新都重新加载模型，速度更快
def initialize_system():
    # A. 检查 data 文件夹是否存在
    if not os.path.exists("data"):
        os.makedirs("data")
        st.warning("⚠️ 'data' 文件夹为空！请放入 .txt 文件后刷新页面。")
        # 创建一个示例文件防止报错
        with open("data/sample.txt", "w", encoding='utf-8') as f:
            f.write("故宫位于北京中心，是明清两代的皇宫。北京是中国的首都。")
    
    # B. 加载模型 (关键点：换成中文模型 BAAI/bge-small-zh-v1.5)
    # 第一次运行会自动下载模型，可能需要一点时间
    st.info("正在加载中文嵌入模型 (BAAI/bge-small-zh-v1.5)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    # C. 读取 data 文件夹下所有 txt 文件
    loader = DirectoryLoader('data/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    
    if not documents:
        return None, None

    # D. 切分文档 (Text Splitting)
    # 把长文章切成 200 字一段，方便检索定位
# 改进版：加入中文标点符号支持，并稍微加大分块大小
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,        # 把块大小从200增加到300，保证能包含更多上下文
        chunk_overlap=50,      # 重叠部分，防止上下文丢失
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""] # 优先级：先按段落切，再按句号切
    )
    splits = text_splitter.split_documents(documents)
    
    # E. 建立向量索引 (Retrieval Module)
    vector_db = FAISS.from_documents(splits, embeddings)
    
    return vector_db, documents

# --- 3. 初始化系统 ---
vector_db, raw_docs = initialize_system()

# 侧边栏显示信息
with st.sidebar:
    st.header("📚 知识库状态")
    if raw_docs:
        st.success(f"已加载 {len(raw_docs)} 篇文章")
        st.write("文件列表:")
        for doc in raw_docs:
            st.code(doc.metadata['source'].split('/')[-1]) # 只显示文件名
    else:
        st.error("未找到文档，请在 data 文件夹中添加 txt 文件。")

# --- 4. 问答交互区域 ---
# 输入框 (Input Query)
query = st.text_input("请输入你的问题：", placeholder="例如：故宫是哪个朝代建立的？")

if query and vector_db:
    # 检索逻辑 (Retrieval)
    # k=3 表示找最相似的 3 个段落
    results = vector_db.similarity_search(query, k=3)
    
    st.markdown("### 🔍 找到的答案段落：")
    
    # 展示结果
    for i, doc in enumerate(results):
        with st.expander(f"参考来源 {i+1} (点击展开/收起)"):
            st.markdown(f"**内容:** {doc.page_content}")
            st.caption(f"来源文件: {doc.metadata['source']}")
            
    # 这里其实完成了 Retrieve (检索)，你可以把最上面的结果当作即时答案
    st.success(f"最佳答案可能是：\n\n{results[0].page_content}")

elif not vector_db:
    st.write("请先准备数据。")