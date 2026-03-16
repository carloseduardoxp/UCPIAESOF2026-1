from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

code1 = "int sum(int a,int b){return a+b;}"
code2 = "int addNumbers(int x,int y){return x+y;}"

inputs1 = tokenizer(code1, return_tensors="pt", truncation=True)
inputs2 = tokenizer(code2, return_tensors="pt", truncation=True)

with torch.no_grad():
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

emb1 = outputs1.last_hidden_state[:,0,:]
emb2 = outputs2.last_hidden_state[:,0,:]

similarity = F.cosine_similarity(emb1, emb2)

print(similarity.item())