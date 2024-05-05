import pandas as pd
from flask import Flask, render_template, request, url_for, redirect
import streamlit as st
import pickle
import requests
import pandas as pd

# Definimos una función llamada 'fetch_poster' que toma un 'movie_id' como argumento.
def fetch_poster(movie_id):
    # Construimos la URL de la API utilizando el 'movie_id' y una clave API válida.
    url = "https://api.themoviedb.org/3/movie/{}?api_key=3d64c9680ddd614f5ec34ba663b30ba6&language=en-US".format(movie_id)
    
    # Hacemos una solicitud GET a la URL y almacenamos la respuesta en la variable 'data'.
    data = requests.get(url)
    
    # Convertimos la respuesta JSON en un diccionario utilizando el método 'json()'.
    data = data.json()
    
    # Extraemos la ruta del póster del diccionario 'data'.
    poster_path = data['poster_path']
    
    # Construimos la URL completa del póster utilizando la base URL de la API y la ruta del póster.
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    
    # Devolvemos la URL completa del póster.
    return full_path

# La función 'fetch_poster' se utiliza para obtener la URL del póster de una película utilizando su 'movie_id'.
# La URL se construye a partir de la información proporcionada por la API 'themoviedb.org'.

# Supongamos que 'movies' es una tabla de datos que contiene información sobre películas,
# y 'similarity' es una matriz de similitud previamente calculada.
# Además, supongamos que tienes la función 'fetch_poster' definida como se mostró anteriormente.

# Definimos una función llamada 'recommend' que toma el nombre de una película como argumento.
def recommend(movie):
    # Buscamos el índice de la película en la tabla 'movies' utilizando su título.
    index = movies[movies['title'] == movie].index[0]

    # Calculamos las distancias de similitud entre la película dada y todas las demás películas.
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    # Listas para almacenar nombres de películas recomendadas y URLs de pósters.
    recommended_movie_names = []
    recommended_movie_posters = []

    # Iteramos sobre las primeras 5 películas más similares (excluyendo la película dada).
    for i in distances[1:6]:
        # Obtenemos el 'movie_id' de la película recomendada.
        movie_id = movies.iloc[i[0]].movie_id
        
        # Usamos la función 'fetch_poster' para obtener la URL del póster de la película.
        recommended_movie_posters.append(fetch_poster(movie_id))
        
        # Agregamos el título de la película recomendada a la lista.
        recommended_movie_names.append(movies.iloc[i[0]].title)

    # Devolvemos las listas de nombres y URLs de pósters de películas recomendadas.
    return recommended_movie_names, recommended_movie_posters


with st.sidebar:
    st.header("CineBot: Tu Recomendador de Películas")
    st.image("https://estaticos-cdn.prensaiberica.es/clip/d2da2628-3686-4561-b430-e7d47ed19fec_16-9-discover-aspect-ratio_default_0.jpg")

    st.header("Proyecto de Recomendación")
    st.write("Echaremos un vistazo en como un poco de codigo y ayuda de la inteligencía artificial, podemos conseguir .")
    st.write("Alumno: Kevin Vallecillo")
    st.write("Correo: kevina.valalm@jcyl.educa.es")

st.header('Sistema de recomendación')
movies = pd.read_pickle('model/movie_list.pkl')
similarity = pd.read_pickle('model/similarity.pkl')

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Selecciona una película de la lista",
    movie_list
)

import streamlit as st


if st.button('Mostrar Recomendaciones'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)

    cols = st.columns(5)  # Cambia a 'st.beta_columns' si estás usando una versión de Streamlit anterior a 0.68.

    for i, col in enumerate(cols):
        col.text(recommended_movie_names[i])
        col.image(recommended_movie_posters[i])

