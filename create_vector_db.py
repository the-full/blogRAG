from omegaconf import OmegaConf
from pymilvus import MilvusClient

from TinyRAG.utils import ReadFiles
from TinyRAG.Embeddings import ZhipuEmbedding


# NOTE: 注意，每次创建都要调一遍智谱的 API，要花钱的！

# 读取配置文件
cfg = OmegaConf.load('./config.yml')

# NOTE: 创建数据库 (如果存在则重建)
client = MilvusClient(cfg.milvus.db_name + ".db")
if client.has_collection(collection_name=cfg.milvus.collection.name):
    client.drop_collection(collection_name=cfg.milvus.collection.name)

client.create_collection(
    collection_name=cfg.milvus.collection.name,
    dimension=cfg.vec_dimension,  # The vectors we will use in this demo has 768 dimensions
)

docs = ReadFiles(cfg.milvus.collection.docs).get_content(max_token_len=600, cover_content=150) # 获得data目录下的所有文件内容并分割
embedding_model = ZhipuEmbedding(dimensions=cfg.vec_dimension) # 创建 EmbeddingModel
vectors = embedding_model(docs) # 将文档编码为词向量

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": 'blog'}
    for i in range(len(vectors))
]


res = client.insert(collection_name=cfg.milvus.collection.name, data=data)
print(f"插入了 {res['insert_count']} 篇文档，ID 为: {res['ids']}。操作耗时: {res['cost']} 毫秒")

