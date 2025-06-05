from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Carrega o dataset MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Converte rótulos para inteiros
y = y.astype(np.uint8)

# Normaliza os pixels
X = X / 255.0

# Separa um subconjunto (KNN é lento com muitos dados)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=7000, test_size=2000, random_state=42)

# Define o classificador KNN
knn = KNeighborsClassifier(n_neighbors=3)

# "Treina" (na prática, armazena os dados)
knn.fit(X_train, y_train)

# Faz predições
y_pred = knn.predict(X_test)

# Avalia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do KNN: {accuracy:.4f}")
