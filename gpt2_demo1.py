from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 选择模型和tokenizer
model_name = "/home/Q21301198/NLP/GPT/pretrain_model/gpt-2"  # 也可以选择其他GPT-2模型，如"gpt2-medium", "gpt2-large", "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 输入文本
text = "Hello, how are you doing today?"

# 使用 tokenizer 编码文本
input_ids = tokenizer.encode(text, return_tensors="pt")

# 生成文本
output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 打印生成的文本
print("Generated Text:")
print(generated_text)