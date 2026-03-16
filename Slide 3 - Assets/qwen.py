from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

code = """
public int calc(int a,int b,int c){
    if(a>0){
        if(b>0){
            return a+b+c;
        }else{
            return a+c;
        }
    }else{
        return c;
    }
}
"""

prompt = f"Refactor this Java code to improve readability:\n{code}"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))