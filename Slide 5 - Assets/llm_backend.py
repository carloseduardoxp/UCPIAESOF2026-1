from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

client = Groq(api_key="sua chave")

@app.route("/llm/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt")

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Você é um desenvolvedor Java altamente eficiente em refactoring. "
                "Você vai receber códigos Java pedindo para melhorar a legibilidade. "
                "Responda com apenas o código melhorado, sem explicações"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False  
    )

    resposta = completion.choices[0].message.content

    return jsonify({"response": resposta})

if __name__ == "__main__":
    app.run(debug=True)