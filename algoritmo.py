import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

# Carregar os dados dos arquivos CSV
iris = pd.read_csv("iris_limpo.csv")
diabetes = pd.read_csv("diabetes_limpo.csv")

print("Antes da remoção de outliers - Iris:")
print(iris.describe())
print("Antes da remoção de outliers - Diabetes:")
print(diabetes.describe())

# Remover colunas categóricas para análise não supervisionada
X_iris = iris.drop(columns=["species"], errors="ignore")
X_diabetes = diabetes.drop(columns=["Outcome"], errors="ignore")

# Normalização dos dados
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)
X_diabetes_scaled = scaler.fit_transform(X_diabetes)

# Função para encontrar o melhor número de clusters
def find_optimal_k(X):
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    kl = KneeLocator(K, inertia, curve="convex", direction="decreasing")
    return kl.elbow

# Encontrar k ótimo para ambas as bases
k_iris = find_optimal_k(X_iris_scaled)
k_diabetes = find_optimal_k(X_diabetes_scaled)
print(f"Número ótimo de clusters para Iris: {k_iris}")
print(f"Número ótimo de clusters para Diabetes: {k_diabetes}")

# Aplicar algoritmos de clusterização
cluster_models_iris = {
    "K-Means": KMeans(n_clusters=k_iris, random_state=42, n_init=10),
    "Bi-Secting K-Means": BisectingKMeans(n_clusters=k_iris, random_state=42),
    "Hierarchical (Linkage)": AgglomerativeClustering(n_clusters=k_iris),
}

cluster_models_diabetes = {
    "K-Means": KMeans(n_clusters=k_diabetes, random_state=42, n_init=10),
    "Bi-Secting K-Means": BisectingKMeans(n_clusters=k_diabetes, random_state=42),
    "Hierarchical (Linkage)": AgglomerativeClustering(n_clusters=k_diabetes),
}

# Função para avaliar e visualizar os clusters
def apply_clustering(X, models, dataset_name):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(15, 5))
    for i, (name, model) in enumerate(models.items()):
        clusters = model.fit_predict(X)
        silhouette = silhouette_score(X, clusters)

        plt.subplot(1, 3, i + 1)
        sns.scatterplot(
            x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="viridis", legend=None
        )
        plt.title(f"{name} - Silhouette: {silhouette:.2f}")

    plt.suptitle(f"Clusters na {dataset_name}")
    plt.show()

# Aplicar e visualizar os clusters na base Iris
apply_clustering(X_iris_scaled, cluster_models_iris, "Base Iris")

# Aplicar e visualizar os clusters na base Diabetes
apply_clustering(X_diabetes_scaled, cluster_models_diabetes, "Base Diabetes")
