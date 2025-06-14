# TFG – Evaluación Comparativa de Métodos Tradicionales y Avanzados en Sistemas de Recomendación

Este repositorio contiene el Trabajo de Fin de Grado titulado:

**"Evaluación Comparativa de Métodos Tradicionales y Avanzados en la Eficiencia y Calidad de Sistemas de Recomendación"**

El objetivo principal es implementar y analizar distintos enfoques de recomendación, tanto clásicos como modernos, evaluando su rendimiento y comportamiento bajo un marco común de métricas.

## Métodos implementados

| Enfoque     | Método                                 | Implementación                       |
|-------------|----------------------------------------|--------------------------------------|
| Tradicional | Filtrado colaborativo (usuario-usuario) | `Surprise`                          |
| Tradicional | Filtrado colaborativo (ítem-ítem)      | `Surprise`                          |
| Tradicional | Factorización matricial (SVD)          | `PyTorch`                           |
| Tradicional | Filtrado basado en contenido (MLP)     | `PyTorch`  |
| Avanzado    | Red Neuronal Multicapa (MLP)           | `PyTorch`                           |
| Avanzado    | Modelo LLM (OpenP5)                    | `HuggingFace` + `OpenP5`            |

---

## Estructura del repositorio
````bash
TFG-Justin/
├── datasets/ # Datos filtrados y embeddings
├── src/ # Implementación de modelos
│ ├── modelos_tradicionales/
│ ├── modelos_avanzados/
│ └── evaluation/
├── results/ # Predicciones, métricas y gráficas
├── utils/ # Métricas personalizadas y utilidades
````
##  Descarga de datasets
Para descargar los datasets se ha de acceder a esta página web: https://grouplens.org/datasets/movielens/ 
##  Ejecución

Cada modelo se ejecuta de forma independiente desde `src/`. Ejemplos:

```bash
python src/modelos_tradicionales/filtrado_uu/train_knn_uu.py
python src/modelos_avanzados/mlp/train_mlp_embeddings.py
Las evaluaciones y métricas están centralizadas en src/evaluation/.
````
Resultados
Los resultados de cada modelo se almacenan en:

results/metrics/: métricas numéricas (RMSE, MAE, R², etc.)

results/graficas/: visualizaciones (bar charts, radar charts, etc.)

Se han empleado métricas tanto clásicas como semánticas y de cobertura/diversidad.

Requisitos
Python 3.10+

Bibliotecas: pandas, numpy, torch, sentence-transformers, surprise, matplotlib, transformers, scikit-learn, etc.
