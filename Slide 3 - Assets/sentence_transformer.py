from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

code1 = "int sum(int a,int b){return a+b;}"
code2 = "int addNumbers(int x,int y){return x+y;}"

emb1 = model.encode(code1)
emb2 = model.encode(code2)

similarity = util.cos_sim(emb1, emb2)

print(similarity)