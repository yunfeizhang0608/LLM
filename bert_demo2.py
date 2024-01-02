from transformers import BertTokenizer, BertForMaskedLM
import torch

# 加载预训练的BERT模型和tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# 输入文本
text = "The quick brown fox [MASK] over the lazy dog."

# 使用tokenizer编码文本
tokens = tokenizer.tokenize(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))
masked_index = tokens.index('[MASK]')
input_ids = tokenizer.convert_tokens_to_ids(tokens)

# 将输入转为PyTorch张量
input_ids = torch.tensor([input_ids])

# 使用BERT进行mask预测
with torch.no_grad():
    outputs = model(input_ids)

# 获取预测结果
predictions = outputs.logits  # predictions的shape为(batch_size, seq_len, vocab_size)

# 提取被mask位置的预测概率分布
masked_token_predictions = predictions[0, masked_index]

# 获取最可能的token的索引
predicted_index = torch.argmax(masked_token_predictions).item()

# 将索引转为token
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

# 打印结果
print("Original Text:", text)
print("Predicted Token:", predicted_token)

