from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from recommender import get_top_recommendations

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Cargar los datos desde el archivo CSV
df = pd.read_csv('stackoverflow_python_questions.csv')

# Endpoint para obtener todos los posts
@app.route('/posts', methods=['GET'])
def get_posts():
    posts = df.to_dict(orient='records')
    return jsonify(posts)

# Endpoint para obtener recomendaciones de los temas más populares basados en el título
@app.route('/recommendations/<int:row_num>', methods=['GET'])
def get_recommendations(row_num):
    if 0 <= row_num < len(df):
        recommendations = get_top_recommendations(df, row_num, top_n=10)
        return jsonify(recommendations)
    else:
        return jsonify({"error": "Invalid row number"}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("80"), debug=True)
