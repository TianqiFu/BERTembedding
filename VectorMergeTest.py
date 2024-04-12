# this experiment purpose is merge 3 vector who sentences vectors, ABSA triple and average_distances_to_centroids
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

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

for t in tqdm(process_list):
    # 对文本进行向量化
    tokens = tokenizer.tokenize(t)
    ids = tokenizer.convert_tokens_to_ids(t)
    input_ids = torch.tensor([ids])

    # 输出文本的向量
    outputs = model(input_ids)
    token_embeddings = outputs[0]

    # 获取句向量
    sentence_embedding = token_embeddings.mean(dim=0)
    print(sentence_embedding)