from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/texto/maiusculo", methods=["POST"])
def maiusculo():
    data = request.json
    texto = data.get("texto")

    resultado = texto.upper()

    return jsonify({"resultado": resultado})


if __name__ == "__main__":
    app.run(debug=True)