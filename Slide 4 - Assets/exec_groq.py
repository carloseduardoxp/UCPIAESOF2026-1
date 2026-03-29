from groq import Groq

client = Groq(api_key="Coloque seu token. Não vai usar o meu")

completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {   "role": "system", 
            "content": "Você é um professor de programação didático e objetivo. Responda com apenas o código solicitado, sem explicações"},
        {
            "role": "user",
            "content": "faz uma função python que calcula fatorial"
        }
    ],

    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=True
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")