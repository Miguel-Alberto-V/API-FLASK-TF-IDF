from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def get_top_recommendations(df, row_num, top_n=10):
    # Usar solo los títulos para el análisis
    df['text'] = df['Title']
    
    # Inicializar TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    # Convertir el texto en una matriz TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
    
    # Calcular la similitud de coseno entre los títulos
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Enumerar los pares (índice, similitud) para el documento en la fila row_num
    similar_posts = list(enumerate(similarity_matrix[row_num]))
    
    # Ordenar los posts similares por la similitud en orden descendente
    sorted_similar_posts = sorted(similar_posts, key=lambda x: x[1], reverse=True)[:top_n + 1]  # +1 para excluir el post en sí mismo
    
    # Devolver los títulos de los posts recomendados
    recommendations = [df.iloc[item[0]]['Title'] for item in sorted_similar_posts if item[0] != row_num]
    
    return recommendations
