# this experiment purpose is merge 3 vector who sentences vectors, ABSA triple and average_distances_to_centroids
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import numpy as np

# 加载BERT模型
model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 准备文本数据
text = "This new version of Things has an entirely different aesthetic from Things 2. Things 2 is much more minimalist; Things 3 seems to have a lot of UI bloat. Not quite sure where the design award came from."
triple ="new version TRUE positive"
average_distances_to_centroids = [8.352705708171774, 8.268810668393852,	8.185850051847236,\
                                  8.325170052401182, 8.255136229996403,	8.211615801028557,\
                                  8.217085891079746, 8.215596766800612,	8.241462068948616,\
                                  8.293398840348594]
process_list = [text, triple]

embedding_list = []
for t in tqdm(process_list):
    # 对文本进行向量化
    tokens = tokenizer.tokenize(t)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([ids])

    # 输出文本的向量
    outputs = model(input_ids)
    token_embeddings = outputs[0]
    print(token_embeddings.shape)

    # 获取句向量
    sentence_embedding = token_embeddings.mean(dim=0)
    embedding_list.append(sentence_embedding)

    # print(sentence_embedding)
    # print(embedding_list)

# 使用平均池化，将两个不同长度的文本向量拼接在一起
mean_tensor1 = torch.mean(embedding_list[0], dim=0)  # 结果形状为[768]
mean_tensor2 = torch.mean(embedding_list[1], dim=0)  # 结果形状为[768]

# 拼接池化后的结果
concatenated_vector = torch.cat((mean_tensor1, mean_tensor2), dim=0)  # 结果形状为[1536]
print(concatenated_vector.shape, concatenated_vector.dtype)

# 将两个向量拼接, 将平均中心距离由list转化为向量
average_distances_to_centroids_tensor = torch.tensor(average_distances_to_centroids, dtype=torch.float32)
extended_vector = torch.cat((concatenated_vector, average_distances_to_centroids_tensor), dim=0)  # 结果形状为[1546]
print(extended_vector.shape)


