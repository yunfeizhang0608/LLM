from transformers import BertTokenizer

# 预训练的BERT模型和分词器
model_name = "/home/Q21301198/NLP/Bert/pretrain_model/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# 设置最大长度
max_len = 512

# 示例文本
# text = "During the past few decades, cognitive diagnostics modeling has attracted increasing attention in computational education communities, which is capable of quantifying the learning status and knowledge mastery levels of students. Indeed, the recent advances in neural networks have greatly enhanced the performance of traditional cognitive diagnosis models through learning the deep representations of students and exercises. Nevertheless, existing approaches often suffer from the issue of overconfidence in predicting students’ mastery levels, which is primarily caused by the unavoidable noise and sparsity in realistic student-exercise interaction data, severely hindering the educational application of diagnostic feedback. To address this, in this paper, we propose a novel Reliable Cognitive Diagnosis (ReliCD) framework, which can quantify the confidence of the diagnosis feedback and is flexible for different cognitive diagnostic functions. Specifically, we first propose a Bayesian method to explicitly estimate the state uncertainty of different knowledge concepts for students, which enables the confidence quantification of diagnostic feedback. In particular, to account for potential differences, we suggest modeling individual prior distributions for the latent variables of different ability concepts using a pretrained model. Additionally, we introduce a logical hypothesis for ranking confidence levels. Along this line, we design a novel calibration loss to optimize the confidence parameters by modeling the process of student performance prediction. Finally, extensive experiments on four real-world datasets clearly demonstrate the effectiveness of our ReliCD framework."

text = "During the past few decades, cognitive diagnostics modeling has attracted increasing attention in computational education communities, which is capable of quantifying the learning status and knowledge mastery levels of students."

# 使用tokenizer对文本进行编码
tokens = tokenizer.encode_plus(
    text,
    max_length=max_len,  # 设置最大长度
    truncation=True,     # 启用截断
    padding="max_length", # 填充到最大长度
    return_tensors="pt"   # 返回PyTorch张量
)

# 输出处理后的结果
# print(tokenizer.vocab)
# print("Original Text:", text)
print('Tokens:', tokens)
print("Tokenize:", tokens['input_ids'].shape)
# print("Tokenized Text:", tokenizer.convert_ids_to_tokens(tokens['input_ids'].squeeze().tolist()))
# print("Truncated and Padded Input IDs:", tokens['input_ids'])
# print("Attention Mask:", tokens['attention_mask'])
