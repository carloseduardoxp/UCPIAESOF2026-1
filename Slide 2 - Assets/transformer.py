from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def similaridade_par_a_par(lista_a, lista_b, imprimir=True):
    n = min(len(lista_a), len(lista_b))
    if n == 0:
        return []

    a = [str(x) for x in lista_a[:n]]
    b = [str(x) for x in lista_b[:n]]

    emb_a = model.encode(a, convert_to_tensor=True)
    emb_b = model.encode(b, convert_to_tensor=True)

    resultados = []
    for i in range(n):
        score = util.pytorch_cos_sim(emb_a[i], emb_b[i]).item()
        resultados.append({
            "i": i + 1,
            "nome_a": a[i],
            "nome_b": b[i],
            "similaridade": score
        })

        if imprimir:
            print(f"{i+1:02d}) A: {a[i]} | B: {b[i]} | sim = {score:.4f}")

    if imprimir and len(lista_a) != len(lista_b):
        print(f"\n[AVISO] Tamanhos diferentes: len(A)={len(lista_a)} vs len(B)={len(lista_b)}. "
              f"Comparei apenas {n} pares.")

    return resultados


# EXEMPLO DE USO
lista_1 = ["getUserName", "calcTotal", "isValidEmail", "fetchData", "renderUI"]
lista_2 = ["retrieveUserName", "computeTotal", "validateEmail", "loadData", "drawUI"]

similaridade_par_a_par(lista_1, lista_2)