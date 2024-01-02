from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载预训练的BERT模型和tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 输入两个句子
sentence1 = "The quick brown fox jumps over the lazy dog."
sentence2 = "A lazy dog is jumped over by a quick brown fox."

# 使用tokenizer编码两个句子
encoding = tokenizer(sentence1, sentence2, return_tensors='pt', truncation=True, padding=True)

# 通过模型进行分类，输出为 logits
with torch.no_grad():
    logits = model(**encoding).logits

# 获取预测结果
probs = torch.nn.functional.softmax(logits, dim=1)
predicted_class = torch.argmax(probs).item()

# 打印结果
print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Predicted Class:", predicted_class)
