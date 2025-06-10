from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# Parâmetros
train_sizes = [5000, 7000, 10000, 12000]
test_sizes = [2000, 3000, 4000]
n_neighbors_list = [3, 5, 7]
n_components_list = [30, 50, 100, 150]  
distance_metrics = ['euclidean', 'manhattan']
random_state = 42

# Carrega e prepara o dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data / 255.0, mnist.target.astype(np.uint8)

# Armazena resultados
results = []

for train_size in train_sizes:
    for test_size in test_sizes:
        for n_components in n_components_list:
            # Redução de dimensionalidade
            pca = PCA(n_components=n_components, random_state=random_state)
            X_pca = pca.fit_transform(X)

            # Divisão dos dados
            X_train, X_test, y_train, y_test = train_test_split(
                X_pca, y, train_size=train_size, test_size=test_size, random_state=random_state)

            # Classificador Linear
            clf = SGDClassifier(random_state=random_state)
            clf.fit(X_train, y_train)
            y_pred_sgd = clf.predict(X_test)
            acc_sgd = accuracy_score(y_test, y_pred_sgd)

            for metric in distance_metrics:
                for n_neighbors in n_neighbors_list:
                    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
                    knn.fit(X_train, y_train)
                    y_pred_knn = knn.predict(X_test)
                    acc_knn = accuracy_score(y_test, y_pred_knn)

                    results.append({
                        'train_size': train_size,
                        'test_size': test_size,
                        'n_components': n_components,
                        'n_neighbors': n_neighbors,
                        'metric': metric,
                        'accuracy_knn': acc_knn,
                        'accuracy_sgd': acc_sgd
                    })

# Salva resultados
df = pd.DataFrame(results)
df.to_csv("resultados_mnist.csv", index=False)

print("Experimentos concluídos. Resultados salvos em 'resultados_mnist.csv'.")