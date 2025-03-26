import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Carregar os dados dos arquivos CSV
iris = pd.read_csv("iris_limpo.csv")
diabetes = pd.read_csv("diabetes_limpo.csv")

# Remover colunas categóricas para análise não supervisionada
X_iris = iris.drop(columns=["species"], errors="ignore")
X_diabetes = diabetes.drop(columns=["Outcome"], errors="ignore")

# Normalização dos dados
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)
X_diabetes_scaled = scaler.fit_transform(X_diabetes)

def plot_dendrogram_interactive(data, dataset_name="Dataset", method="complete"):
    """
    Função para gerar e plotar o dendrograma interativo de uma base de dados.
    
    Parâmetros:
        data: array com os dados (normalizados, se necessário)
        dataset_name: nome da base de dados (usado no título do gráfico)
        method: método de linkage a ser utilizado (ex.: 'single', 'complete', 'average')
    """
    # Gerar a matriz de linkage utilizando o método especificado
    Z = linkage(data, method=method)
    
    # Plotar o dendrograma
    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(Z, ax=ax)
    plt.title(f'Dendrograma - {dataset_name} (Linkage: {method})')
    plt.xlabel('Amostras')
    plt.ylabel('Distância')

    # Função para capturar o clique e exibir o corte
    def on_click(event):
        if event.inaxes == ax:
            cutoff = event.ydata
            plt.axhline(y=cutoff, color='r', linestyle='--')
            clusters = fcluster(Z, t=cutoff, criterion='distance')
            print(f"Corte na altura {cutoff:.2f} resultou em {len(set(clusters))} clusters.")
            plt.draw()

    # Conectar o evento de clique
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

# Gerar e exibir o dendrograma interativo para a base Iris
plot_dendrogram_interactive(X_iris_scaled, dataset_name="Iris", method="complete")

# Gerar e exibir o dendrograma interativo para a base Diabetes
plot_dendrogram_interactive(X_diabetes_scaled, dataset_name="Diabetes", method="complete")
