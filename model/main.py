import numpy as np
import pandas as pd
import ast

pd.set_option('display.max_columns', None)

movies = pd.read_csv('/content/tmdb_5000_movies.csv')
credits = pd.read_csv('/content/tmdb_5000_credits.csv', encoding='utf-8')

print(movies.shape)
movies.head(1)

print(credits.shape)
credits.head(1)

movies = movies.merge(credits, on='title')
movies.dropna(inplace=True)

movies.head(2)

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

import ast  # Importamos el módulo 'ast' que nos permitirá analizar y evaluar código Python en forma de árbol sintáctico abstracto.

def convert(text):
    L = []  # Creamos una lista vacía llamada 'L' para almacenar los valores extraídos del diccionario.
    for i in ast.literal_eval(text):  # Iteramos a través de los elementos obtenidos al evaluar el texto como código Python.
        L.append(i['name'])  # Extraemos el valor correspondiente a la clave 'name' en el diccionario 'i' y lo agregamos a la lista 'L'.
    return L  # Devolvemos la lista que contiene los valores 'name' extraídos de cada diccionario.

# Nota: Es importante destacar que este código asume que 'text' contiene una representación válida de una lista de diccionarios.
# La función 'ast.literal_eval' se utiliza para evaluar expresiones literales de Python de manera segura.
# Si 'text' no es una lista válida de diccionarios, este código podría generar excepciones.

# Se asume que 'movies' es un DataFrame que contiene información sobre películas, y 'genres' es una columna que contiene listas de diccionarios.
# La intención es aplicar la función 'convert' a cada elemento de la columna 'genres'.

# Se aplica la función 'convert' a cada elemento de la columna 'genres/keywords' utilizando el método 'apply'.
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Después de aplicar la función 'convert' a cada elemento de la columna 'genres/keywords',
# se espera que la columna 'genres' contenga listas de nombres extraídos de los diccionarios originales.

movies.head(2)

import ast
ast.literal_eval("[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}, {'id': 14, 'name': 'Fantasy'}, {'id': 878, 'name': 'Science Fiction'}]")

# Se define una función llamada 'convert3' que toma un argumento 'text'.
def convert3(text):
    L = []  # Se crea una lista vacía llamada 'L' para almacenar los valores extraídos del diccionario.
    counter = 0  # Se inicializa un contador en 0 para rastrear la cantidad de elementos procesados.
    
    # Se itera a través de los elementos obtenidos al evaluar el texto como código Python.
    for i in ast.literal_eval(text):
        if counter < 3:  # Se verifica si el contador es menor que 3.
            L.append(i['name'])  # Si el contador es menor que 3, se extrae el valor de la clave 'name' y se agrega a la lista 'L'.
            counter += 1  # Se incrementa el contador en 1 en cada iteración.
    
    return L  # Se devuelve la lista que contiene los valores 'name' extraídos de los primeros tres diccionarios.

# Esta función 'convert3' está diseñada para extraer los nombres de los primeros tres elementos en una lista de diccionarios.
# Si el contador excede 3, los elementos restantes no se agregarán a la lista 'L'.

movies['cast'] = movies['cast'].apply(convert3)
movies.head(2)

# La función lambda toma cada elemento 'x' de la columna 'cast' y toma solo los primeros 3 elementos de esa lista.
# Esto cortará la lista del elenco para incluir solo los primeros 3 elementos.
movies['cast'] = movies['cast'].apply(lambda x: x[:3])
movies.head(2)

# Definimos una función llamada 'fetch_director' que toma un argumento 'text'.
def fetch_director(text):
    L = []  # Creamos una lista vacía llamada 'L' para almacenar los nombres de los directores.
    # Iteramos a través de los elementos obtenidos al evaluar el texto como código Python.
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':  # Verificamos si el valor de la clave 'job' es igual a 'Director'.
            L.append(i['name'])  # Si es así, agregamos el valor de la clave 'name' a la lista 'L'.
    return L  # Devolvemos la lista que contiene los nombres de los directores.

# La función 'fetch_director' está diseñada para extraer los nombres de los directores de una lista de diccionarios.
# La función verifica si el trabajo ('job') de una persona en la lista es 'Director' y solo agrega esos nombres a la lista 'L'.

# Aplicamos la función 'fetch_director' a la columna 'crew' de la tabla 'movies' utilizando el método 'apply'.
# Esto transformará los datos en la columna 'crew' y extraerá los nombres de los directores.
movies['crew'] = movies['crew'].apply(fetch_director)
movies.sample(2)

# Aplicamos una operación a la columna 'overview' de la tabla 'movies' utilizando el método 'apply'.
# Usamos una función lambda para dividir cada resumen en palabras individuales.
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies.sample(2)

# Definimos una función llamada 'collapse' que toma una lista 'L' como argumento.
def collapse(L):
    L1 = []  # Creamos una lista vacía llamada 'L1' para almacenar los elementos transformados.
    # Iteramos a través de los elementos en la lista 'L'.
    for i in L:
        L1.append(i.replace(" ", ""))  # Reemplazamos los espacios en blanco por cadenas vacías y agregamos el resultado a 'L1'.
    return L1  # Devolvemos la lista 'L1' que contiene los elementos de 'L' con los espacios eliminados.

# La función 'collapse' está diseñada para eliminar los espacios en blanco de los elementos en una lista 'L'.
# Devuelve una nueva lista con los elementos transformados.

movies['cast'] = movies['cast'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies.head(2)

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies.head(2)

new = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])
new.head(2)

# La función lambda toma cada lista 'x' en la columna 'tags' y utiliza el método 'join' para unir los elementos de la lista con espacios.
# Esto transformará la lista de etiquetas en una cadena de texto única separada por espacios.
new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head(2)

# Importamos la clase 'CountVectorizer' del módulo 'feature_extraction.text' de la librería Scikit-learn.
from sklearn.feature_extraction.text import CountVectorizer

# Creamos una instancia de 'CountVectorizer' con ciertos parámetros.
# 'max_features' limita el número máximo de características (palabras) consideradas en el vectorizador.
# 'stop_words' especifica que se deben ignorar las palabras en inglés comunes (stop words) durante el proceso de vectorización.
cv = CountVectorizer(max_features=5000, stop_words='english')

# El objeto 'cv' ahora es una instancia de la clase 'CountVectorizer' configurado con los parámetros indicados.
# Será utilizado para convertir textos en una representación numérica, específicamente una matriz de conteo de palabras.

# Utilizamos el objeto 'cv' (CountVectorizer) que definiste previamente para transformar la columna 'tags' en una matriz de conteo de palabras.
# El método 'fit_transform' realiza el proceso de transformación y devuelve una matriz dispersa (sparse matrix).
vector = cv.fit_transform(new['tags']).toarray()

# Después de esta línea, 'vector' contendrá la representación numérica de las etiquetas en forma de una matriz de conteo de palabras.
# Cada fila representa un registro (documento) en la tabla 'new', y cada columna representa una palabra única en el conjunto de datos.

vector.shape

# Importamos la función 'cosine_similarity' del módulo 'metrics.pairwise' de la librería Scikit-learn.
from sklearn.metrics.pairwise import cosine_similarity

# Utilizamos la función 'cosine_similarity' para calcular la similitud coseno entre las filas de la matriz 'vector'.
# La matriz 'vector' contiene la representación numérica de las etiquetas en forma de matriz de conteo de palabras.

# La función 'cosine_similarity' calcula la similitud coseno entre cada par de filas (registros) en la matriz 'vector'.
similarity = cosine_similarity(vector)

# Después de esta línea, 'similarity' contendrá una matriz que representa la similitud coseno entre todos los pares de registros.
# Cada valor en la matriz indica qué tan similares son dos registros en función de sus etiquetas convertidas en conteo de palabras.

similarity

new[new['title'] == "The Lego Movie"].index[0] # Esto se supone que es una comparación para localizar una película en el DataFrame.

# Supongamos que 'new' es una tabla de datos que contiene información sobre películas y 'similarity' es una matriz de similitud calculada previamente.

# Definimos una función llamada 'recommend' que toma el nombre de una película como argumento.
def recommend(movie):
    # Buscamos el índice de la película en la tabla 'new' utilizando su título.
    index = new[new['title'] == movie].index[0]

    # Calculamos las distancias entre la película dada y todas las demás películas utilizando la matriz 'similarity'.
    # Las distancias se almacenan en la lista 'distances' junto con los índices de las películas correspondientes.
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    # Iteramos sobre las primeras 5 películas más similares (excluyendo la película dada).
    for i in distances[1:6]:
        # Imprimimos el título de las películas recomendadas basadas en la similitud.
        print(new.iloc[i[0]].title)

# La función 'recommend' muestra los títulos de las 5 películas más similares a la película dada como argumento.
# La similitud se determina según la matriz de similitud previamente calculada.

recommend("The Dark Knight Rises")
# Esto es una llamada a la función 'recommend' que probablemente imprimirá una lista de películas similares a "The Dark Knight Rises".

import pickle
# Importamos el módulo 'pickle', que se utiliza para serializar (guardar) y deserializar (cargar) objetos de Python.

# Guardamos el DataFrame 'new' en un archivo llamado 'movie_list.pkl' utilizando la función 'dump' de 'pickle'.
# El modo 'wb' se usa para escribir en modo binario.
pickle.dump(new, open('movie_list.pkl', 'wb'))

# Guardamos la matriz 'similarity' en un archivo llamado 'similarity.pkl' utilizando la función 'dump' de 'pickle'.
# El modo 'wb' se usa para escribir en modo binario.
pickle.dump(similarity, open('similarity.pkl', 'wb'))

# Estos pasos guardan el DataFrame 'new' y la matriz 'similarity' en archivos pkl (archivos de pickle).
# Esto permite que puedas cargar estos objetos posteriormente sin necesidad de volver a calcularlos.

movies = pickle.load(open('movie_list.pkl', 'rb'))
# Aquí cargamos el DataFrame 'movies' desde un archivo pkl en modo binario 'rb' (read binary).

movies
# Esta línea probablemente imprima o muestre el DataFrame 'movies' que se acaba de cargar desde el archivo 'movie_list.pkl'.
