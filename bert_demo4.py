from transformers import BertTokenizer, BertForQuestionAnswering
import torch

def ask_question(context, question):
    # 加载预训练的BERT模型和分词器
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)

    # 分词和编码输入
    inputs = tokenizer(question, context, return_tensors="pt")
    
    # 进行问答预测
    start_logits, end_logits = model(**inputs).logits.split(1, dim=-1)
    
    # 获取最可能的答案
    start_index = torch.argmax(start_logits, dim=1).item()
    end_index = torch.argmax(end_logits, dim=1).item() + 1
    answer = tokenizer.decode(inputs["input_ids"][0, start_index:end_index])

    return answer

# 示例文本和问题
context = "Hugging Face is a technology company based in New York City."
question = "Where is Hugging Face located?"

# 获取答案
answer = ask_question(context, question)
print(f"Question: {question}")
print(f"Answer: {answer}")

