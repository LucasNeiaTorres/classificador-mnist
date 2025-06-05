# classificador-mnist

## Descrição
Este repositório contém um script classificador (usando kNN e classificador linear) para o dataset MNIST, que é um conjunto de dados de imagens de dígitos manuscritos. O script utiliza a biblioteca `scikit-learn` para treinar e avaliar o modelo. O objetivo é classificar imagens de dígitos manuscritos em suas respectivas classes (0-9). 

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