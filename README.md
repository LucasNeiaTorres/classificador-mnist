# classificador-mnist

> **In English, briefly.** A grid experiment on MNIST handwritten digits: it sweeps
> 288 combinations of training size, test size, number of PCA components, `k` and
> distance metric, and records the accuracy of two classifiers side by side — a
> k-Nearest Neighbors and a linear one (`SGDClassifier`) — into a CSV that is
> committed to the repository. The point is not to reach state-of-the-art accuracy;
> it is to see, on the same axes, where a lazy instance-based method and a linear
> model disagree. They disagree most on PCA: as the number of components grows, kNN
> gets *worse* and the linear model gets *better*. A second script turns the CSV
> into the charts. Python, scikit-learn, no notebook.

Varredura de hiperparâmetros na classificação de dígitos manuscritos do MNIST,
comparando **kNN** e **classificador linear (SGD)** sob as mesmas divisões de dados
e a mesma redução de dimensionalidade. As 288 medições ficam versionadas em
`resultados_mnist.csv`, e os gráficos saem daí.

---

## Contexto: por que isto existe

Trabalho de disciplina, com uma exigência que mudou o formato do código: o
resultado precisava virar **relatório científico**, com gráficos e uma comparação
defensável entre dois classificadores — não um número solto de acurácia.

Isso empurra o projeto para longe do "treina um modelo e mostra o score". Se a
comparação tem que sustentar um argumento, ela precisa de uma **grade completa**:
os dois classificadores vendo exatamente os mesmos dados, sob exatamente a mesma
transformação, em várias condições. E o resultado precisa ficar guardado em formato
que se possa reprocessar sem treinar tudo de novo.

## A decisão de projeto: separar medir de analisar

O repositório tem dois scripts, e a separação é deliberada.

| Script | Papel |
|---|---|
| `classificadorMNIST.py` | **Mede.** Roda a grade inteira e escreve `resultados_mnist.csv`. É caro. |
| `analyseMetrics.py` | **Analisa.** Lê o CSV e gera os gráficos. É barato e se roda quantas vezes quiser. |

Treinar 288 kNNs sobre o MNIST não é algo que se queira repetir porque o rótulo de
um eixo ficou feio. Com o CSV versionado, mudar a pergunta ("e se eu agrupar por
métrica em vez de por `k`?") custa segundos, e qualquer pessoa pode conferir os
números sem rodar nada — eles estão no repositório, não só no gráfico.

## A grade

```python
train_sizes      = [5000, 7000, 10000, 12000]
test_sizes       = [2000, 3000, 4000]
n_components_list= [30, 50, 100, 150]     # PCA
n_neighbors_list = [3, 5, 7]              # k do kNN
distance_metrics = ['euclidean', 'manhattan']
random_state     = 42
```

4 × 3 × 4 × 3 × 2 = **288 linhas** em `resultados_mnist.csv`, cada uma com
`accuracy_knn` e `accuracy_sgd`.

Os pixels são normalizados por `/255.0` e o PCA é aplicado antes da divisão
treino/teste. O `SGDClassifier` é treinado uma vez por combinação de
(`train_size`, `test_size`, `n_components`) — por isso a coluna `accuracy_sgd` se
repete nas seis linhas de `k` × métrica que compartilham a mesma divisão.

## Resultados registrados

Todos os números abaixo são **transcritos de `resultados_mnist.csv`**, tal como
versionado. Nada foi recalculado.

Cabeçalho: `train_size,test_size,n_components,n_neighbors,metric,accuracy_knn,accuracy_sgd`

**Melhor acurácia de kNN da grade:**

```
12000,2000,30,3,euclidean,0.96,0.8665
```

**Pior acurácia de kNN da grade:**

```
5000,3000,150,7,manhattan,0.9063333333333333,0.8816666666666667
```

**Melhor acurácia do classificador linear (SGD):**

```
12000,2000,150,7,euclidean,0.949,0.9035
```

**Pior acurácia do classificador linear (SGD):**

```
5000,2000,30,3,euclidean,0.942,0.8525
```

O padrão que o CSV registra, e que os gráficos mostram: **o kNN ganha do linear em
toda a grade**, mas os dois reagem ao PCA em direções opostas — a acurácia média do
kNN cai conforme o número de componentes sobe, enquanto a do SGD sobe. É o efeito
esperado da maldição da dimensionalidade sobre um método baseado em distância:
componentes a mais trazem ruído que engorda a distância sem trazer sinal, e a
vizinhança piora. O modelo linear, que não mede distância, só ganha com mais
informação.

Entre `euclidean` e `manhattan`, os dois valores de acurácia média ficam próximos,
com vantagem da euclidiana (`grafico_metric.png`).

> Os gráficos são a leitura de médias agregadas por eixo. Os valores exatos de cada
> ponto não estão escritos no repositório — o que está escrito, linha a linha, é o
> CSV.

## Gráficos

Gerados por `analyseMetrics.py` a partir do CSV, um por eixo da grade:

| Arquivo | Eixo |
|---|---|
| `grafico_train_size.png` | tamanho do treino |
| `grafico_test_size.png` | tamanho do teste |
| `grafico_n_components.png` | componentes do PCA |
| `grafico_n_neighbors.png` | `k` do kNN |
| `grafico_metric.png` | métrica de distância |

Há ainda um `grafico_pca_acuracia.png`, que **não é produzido por
`analyseMetrics.py`** e cobre apenas 30, 50 e 100 componentes — é de uma execução
anterior, antes de 150 entrar na grade.

## Como rodar

```sh
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python3 classificadorMNIST.py     # roda a grade → resultados_mnist.csv
python3 analyseMetrics.py         # lê o CSV → grafico_*.png
```

Requer Python 3.8+. O MNIST é baixado na primeira execução por
`fetch_openml('mnist_784')` — não vem no repositório. A grade inteira treina 288
kNNs e 48 modelos lineares; conte minutos, não segundos.

## Limitações conhecidas

- **O PCA é ajustado sobre o dataset inteiro, antes da divisão treino/teste.** Ou
  seja, a transformação viu as amostras de teste. É vazamento de dados — brando,
  porque o PCA não usa os rótulos, mas real: num experimento rigoroso o `fit` do
  PCA iria só no treino, e o teste passaria por `transform`. As acurácias aqui
  tendem a ser levemente otimistas.
- **`requirements.txt` lista `tensorflow`, que nenhum dos dois scripts importa.**
  Sobra de uma versão anterior; instalar é caro e desnecessário.
- **O `SGDClassifier` roda com os padrões**, sem ajuste de `loss`, `alpha` ou
  `max_iter`, e sem padronização das features além do `/255.0` e do PCA. A
  comparação é honesta quanto aos dados vistos, mas o lado linear não foi
  otimizado como o kNN foi.
- **Só acurácia.** Não há matriz de confusão, precisão/revocação por dígito nem
  tempo de inferência — e o tempo seria o eixo em que o kNN perde feio, já que ele
  empurra todo o custo para a predição.
- `random_state=42` fixo em todas as etapas: os números são reproduzíveis, mas
  cada célula da grade é **uma** execução, sem repetição nem desvio-padrão.

## Estrutura

```
.
├── classificadorMNIST.py      # roda a grade de 288 combinações
├── analyseMetrics.py          # CSV → gráficos
├── resultados_mnist.csv       # as 288 medições, versionadas
├── requirements.txt
├── grafico_train_size.png
├── grafico_test_size.png
├── grafico_n_components.png
├── grafico_n_neighbors.png
├── grafico_metric.png
└── grafico_pca_acuracia.png   # de uma execução anterior (sem 150 componentes)
```
