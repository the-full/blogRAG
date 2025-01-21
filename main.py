import socket

from omegaconf import OmegaConf
from pymilvus import MilvusClient

from TinyRAG.LLM import DeepSeekChat 
from TinyRAG.Embeddings import ZhipuEmbedding

# 读取配置文件
cfg = OmegaConf.load('./config.yml')

# NOTE: 连接数据库
client = MilvusClient(cfg.milvus.db_name + ".db")
if client.has_collection(collection_name=cfg.milvus.collection.name):
    client.create_collection(
        collection_name=cfg.milvus.collection.name,
        dimension=cfg.vec_dimension,
    )
else:
    print(f"警告: 集合 {cfg.milvus.collection.name} 不存在")

embedding_model = ZhipuEmbedding(dimensions = cfg.vec_dimension) # 创建EmbeddingModel

def handle_client(conn):
    try:
        question = conn.recv(1024).decode('utf-8').strip()
        if not question:
            return
        
        result = client.search(
            collection_name=cfg.milvus.collection.name,
            data=embedding_model([question]),
            filter="subject == 'blog'",
            limit=cfg.milvus.search_limit,
            output_fields=["text", "subject"],
        )
        if len(result) == 0:
            content = "知识库中未查询到结果"
        else:
            content = (
                "知识库中查询到如下结果:\n"
                + "\n".join([f"{item['entity']['text']}" for item in result[0]])
                + "\n"
            )
        model = DeepSeekChat()
        response = model.chat(question, [], content) + '\n'
        conn.send(response.encode('utf-8'))
    finally:
        conn.close()


def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 7788))
    server.listen(5)
    print("服务已启动，正在监听 7788 端口...")
    
    while True:
        conn, addr = server.accept()
        print(f"收到来自 {addr} 的连接")
        handle_client(conn)

if __name__ == '__main__':
    start_server()
