from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "codellama/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "def factorial(n):"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_length=80)

print(tokenizer.decode(outputs[0]))