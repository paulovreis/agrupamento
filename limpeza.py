import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_diabetes

# Carregar os dados da base Iris
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

# Carregar os dados da base Diabetes
diabetes = load_diabetes()
df_diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Função para limpar os dados
def limpar_base(df, nome_arquivo):
    # 1. Verificar valores ausentes
    print(f"Valores ausentes em {nome_arquivo}:\n", df.isnull().sum())
    df.dropna(inplace=True)
    
    # 2. Remover duplicatas
    df.drop_duplicates(inplace=True)
    
    # 3. Detectar e remover outliers usando o IQR (Intervalo Interquartil)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    df = df[~((df < limite_inferior) | (df > limite_superior)).any(axis=1)]
    
    # 4. Normalizar os dados
    df_normalized = (df - df.min()) / (df.max() - df.min())
    
    # 5. Salvar a base de dados limpa em um arquivo CSV
    df_normalized.to_csv(f"{nome_arquivo}_limpo.csv", index=False)
    print(f"Base {nome_arquivo} salva como {nome_arquivo}_limpo.csv\n")
    
# Aplicar a função de limpeza nas bases de dados
limpar_base(df_iris, "iris")
limpar_base(df_diabetes, "diabetes")
