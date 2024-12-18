import pandas as pd
import numpy as np
from collections import Counter
from math import log

# Función para cargar el dataset
def cargar_dataset(ruta_csv):
    # Saltar la primera fila usando skiprows=1
    df = pd.read_csv(ruta_csv, sep=';', skiprows=1, header=None, names=["id", "text", "date", "label"], low_memory=False)
    
    # Convertir la columna 'label' a tipo int
    df["label"] = pd.to_numeric(df["label"], errors='coerce')  # Coerce convierte valores no numéricos a NaN
    
    # Eliminar filas no válidas si las hay
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)  # Convertir la columna 'label' a entero
    return df



# División balanceada en train y test
def dividir_train_test(df, test_size=0.2, seed=42):
    # Estratificación por etiqueta
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df['label'])
    return train_df, test_df

# Construir diccionario y estadísticas
def construir_diccionario(train_df):
    # Separar textos por clase
    pos_texts = train_df[train_df['label'] == 1]['text']
    neg_texts = train_df[train_df['label'] == 0]['text']

    # Contar palabras por clase
    word_counts_pos = Counter()
    word_counts_neg = Counter()

    for text in pos_texts:
        word_counts_pos.update(text.split())

    for text in neg_texts:
        word_counts_neg.update(text.split())

    # Priors
    total_tweets = len(train_df)
    p_pos = len(pos_texts) / total_tweets
    p_neg = len(neg_texts) / total_tweets

    return {
        'word_counts_pos': word_counts_pos,
        'word_counts_neg': word_counts_neg,
        'p_pos': p_pos,
        'p_neg': p_neg,
        'vocab_size': len(set(word_counts_pos.keys()).union(set(word_counts_neg.keys())))
    }

# Clasificación de un tweet usando Naïve Bayes
def predecir_tweet(tweet, stats, alpha=1):
    log_prob_pos = log(stats['p_pos'])
    log_prob_neg = log(stats['p_neg'])
    vocab_size = stats['vocab_size']

    words = tweet.split()
    for word in words:
        # Laplace smoothing
        p_w_pos = (stats['word_counts_pos'].get(word, 0) + alpha) / (sum(stats['word_counts_pos'].values()) + alpha * vocab_size)
        p_w_neg = (stats['word_counts_neg'].get(word, 0) + alpha) / (sum(stats['word_counts_neg'].values()) + alpha * vocab_size)
        log_prob_pos += log(p_w_pos)
        log_prob_neg += log(p_w_neg)

    return 1 if log_prob_pos > log_prob_neg else 0

# Evaluación del modelo
def evaluar_modelo(test_df, stats):
    y_true = test_df['label'].values
    y_pred = [predecir_tweet(row['text'], stats) for _, row in test_df.iterrows()]

    accuracy = np.mean(y_pred == y_true)
    print("Exactitud del modelo:", accuracy)
    return accuracy

# Main
if __name__ == "__main__":
    ruta_csv = "./CRI_Practica3/FinalStemmedSentimentAnalysisDataset.csv"

    # Cargar datos
    df = cargar_dataset(ruta_csv)
    print("Ejemplo de datos:\n", df.head())

    # División en train y test
    train_df, test_df = dividir_train_test(df)
    print(f"\nTamaño del conjunto de entrenamiento: {len(train_df)}")
    print(f"Tamaño del conjunto de prueba: {len(test_df)}")

    # Construir diccionario y estadísticas
    stats = construir_diccionario(train_df)

    # Evaluar el modelo
    evaluar_modelo(test_df, stats)
