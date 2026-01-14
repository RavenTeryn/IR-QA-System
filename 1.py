from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader


# 1. 准备文档 (假设你有个 data.txt)
# loader = TextLoader("data/your_article.txt") 
# documents = loader.load()

# 为了演示，我们直接用文字代替
docs = ["北京是中国的首都。", "故宫位于北京中心，是明清两代的皇宫。", "上海是中国最大的经济中心。"]

# 2. 切分文档并建立索引 (对应功能要求 2)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_texts(docs, embeddings)

# 3. 检索与输出 (对应功能要求 4)
query = "故宫在哪里？"
# 寻找最相关的 1 个段落
relevant_docs = vector_db.similarity_search(query, k=1) 

print(f"问题: {query}")
print(f"找到的答案段落: {relevant_docs[0].page_content}")