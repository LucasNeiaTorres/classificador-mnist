# Recarregar bibliotecas e o CSV após reset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do arquivo
file_path = "./resultados_mnist.csv"
df = pd.read_csv(file_path)

# Definir estilo
sns.set(style="whitegrid")

# Colunas para análise
colunas = ["train_size", "test_size", "n_components", "n_neighbors", "metric"]

# Gerar e salvar gráficos
output_paths = {}
for coluna in colunas:
    grouped = df.groupby(coluna)[["accuracy_knn", "accuracy_sgd"]].mean().reset_index()
    plt.figure(figsize=(8, 5))

    if df[coluna].dtype == 'object':
        sns.barplot(data=grouped.melt(id_vars=coluna, var_name='classificador', value_name='acuracia'),
                    x=coluna, y='acuracia', hue='classificador')
    else:
        sns.lineplot(data=grouped, x=coluna, y="accuracy_knn", label="kNN", marker="o")
        sns.lineplot(data=grouped, x=coluna, y="accuracy_sgd", label="SGD", marker="o")

    plt.title(f"Acurácia média por {coluna}")
    plt.xlabel(coluna)
    plt.ylabel("Acurácia Média")
    plt.legend()
    plt.tight_layout()

    output_path = f"./grafico_{coluna}.png"
    plt.savefig(output_path)
    plt.close()
    output_paths[coluna] = output_path

output_paths
