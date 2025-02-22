from flask import Flask, request, render_template
import os
import keras
import keras_nlp

# Configurar credenciales de Kaggle (si es necesario)
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

# Cargar el modelo Gemma
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma2_instruct_2b_en")

# Crear la aplicación Flask
app = Flask(__name__)

# Ruta principal
@app.route("/", methods=["GET", "POST"])
def index():
    output = ""
    if request.method == "POST":
        # Obtener el prompt del formulario
        prompt = request.form.get("prompt", "")
        # Generar texto con el modelo
        output = gemma_lm.generate(prompt, max_length=100)
    return render_template("index.html", output=output)

# Iniciar la aplicación
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
