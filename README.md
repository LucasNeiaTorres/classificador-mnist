# classificador-mnist

## Descrição
Este repositório contém um script que realiza a classificação de dígitos manuscritos utilizando o dataset MNIST. Foram aplicados dois classificadores: **k-Nearest Neighbors (kNN)** e **classificador linear (SGDClassifier)**. Além disso, o script realiza:

- Redução de dimensionalidade com **PCA (Principal Component Analysis)**;
- Variações nas métricas de distância (`euclidean`, `manhattan`);
- Diferentes valores de `k` no kNN;
- Múltiplos tamanhos para os conjuntos de treino e teste.

O script gera um arquivo `resultados_mnist.csv` com as acurácias para cada combinação testada, e o script `analyseMetrics.py` gera gráficos para inclusão em relatórios científicos.

## Pré-requisitos
Certifique-se de ter o seguinte instalado em seu sistema:
- Python 3.8 ou superior
- `pip` para gerenciar pacotes Python

## Configuração do Ambiente

1. **Clone o repositório**:
   ```sh
   git clone <URL_DO_REPOSITORIO>
   cd classificador-mnist
    ```
2. **Crie o ambiente virtual**:
   ```sh
   python3 -m venv venv
   ```

3. **Ative o ambiente virtual**:
    - No Windows:
      ```sh
      venv\Scripts\activate
      ```
    - No Linux/Mac:
      ```sh
      source venv/bin/activate
      ```

4. **Instale as dependências**:
    ```sh
    pip install -r requirements.txt
    ```

## Uso
Para executar o script, use o seguinte comando:

```sh
python3 classificadorMNIST.py
```
