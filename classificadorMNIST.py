from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np

train_size = 7000
test_size = 2000
random_state = 42
n_neighbors = 3
n_components = 50  # Reduz de 784 para 50

# Carrega o dataset MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Converte rótulos para inteiros
y = y.astype(np.uint8)

# Normaliza os pixels
X = X / 255.0

# Aplica PCA para redução de dimensionalidade
pca = PCA(n_components=n_components, random_state=random_state)
X = pca.fit_transform(X)

# Separa os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=random_state)

# KNN
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Classificador linear
clf = SGDClassifier(random_state=random_state)
clf.fit(X_train, y_train)
y_pred_sgd = clf.predict(X_test)
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)

# Resultados
print(f"Acurácia dos classificadores com redução PCA para {n_components} dimensões para parâmetros [train_size={train_size}, test_size={test_size}, random_state={random_state}, n_neighbors={n_neighbors}]")
print(f"KNN: {accuracy_knn:.4f}")
print(f"Classificador linear (SGD): {accuracy_sgd:.4f}")