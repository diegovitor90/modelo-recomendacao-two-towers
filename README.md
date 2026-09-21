# Sistema de Recomendação Two-Tower

Projeto acadêmico de Machine Learning que implementa um modelo de recomendação baseado na arquitetura **Two-Tower** para estimar a probabilidade de interação entre usuários e itens.

O modelo utiliza embeddings e duas redes neurais independentes: uma aprende representações dos usuários e a outra aprende representações dos itens. Em seguida, essas representações são comparadas para estimar a relevância ou probabilidade de interação entre cada usuário e cada item.

## Objetivo

Sistemas de recomendação precisam identificar quais itens podem ser mais relevantes para cada usuário com base em interações, características ou comportamentos disponíveis nos dados.

Este projeto explora a arquitetura Two-Tower como uma abordagem para aprender representações de usuários e itens, permitindo estimar interações e apoiar recomendações personalizadas.

## Arquitetura do modelo

O modelo é composto por duas redes neurais:

- **Torre de Usuário:** aprende uma representação vetorial de cada usuário por meio de embeddings e camadas de rede neural.
- **Torre de Item:** aprende uma representação vetorial de cada item por meio de embeddings e camadas de rede neural.
- **Camada de comparação:** combina ou compara as representações de usuário e item para estimar a probabilidade de interação.

```text
Características do usuário → Torre de Usuário ─┐
                                               ├→ Escore de similaridade/interação → Predição
Características do item    → Torre de Item    ─┘
```

## Tecnologias utilizadas

- Python
- Jupyter Notebook
- Pandas
- NumPy
- TensorFlow
- Keras
- Embeddings
- Redes neurais
- Sistemas de recomendação
- Git e GitHub

## Estrutura do repositório

```text
.
├── Two_Tower.ipynb
├── README.md
└── requirements.txt
```

> O arquivo `requirements.txt` é opcional. Caso ele ainda não exista, veja a seção de instalação abaixo.

## Notebook principal

A implementação principal está disponível no arquivo:

```text
Two_Tower.ipynb
```

O notebook contém etapas de exploração, preparação de dados, criação do modelo, treinamento e análise do protótipo de recomendação.

## Como executar localmente

### Pré-requisitos

- Python 3.10 ou superior
- Jupyter Notebook ou JupyterLab
- Git

### 1. Clone o repositório

> Ajuste a URL se o repositório for renomeado.

```bash
git clone [https://github.com/diegovitor90/SQL.git](https://github.com/diegovitor90/SQL.git)
cd SQL
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv
```

### 3. Ative o ambiente virtual

#### Windows

```powershell
venv\Scripts\activate
```

#### Linux ou macOS

```bash
source venv/bin/activate
```

### 4. Instale as dependências

Se houver um arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

Caso ainda não exista:

```bash
pip install pandas numpy tensorflow jupyter
```

### 5. Execute o Jupyter Notebook

```bash
jupyter notebook
```

Depois, abra o arquivo:

```text
Two_Tower.ipynb
```

## Competências desenvolvidas

Neste projeto, pratiquei:

- Manipulação e preparação de dados com Pandas e NumPy
- Organização de dados para problemas de Machine Learning
- Criação e uso de embeddings
- Desenvolvimento de redes neurais com TensorFlow/Keras
- Arquitetura Two-Tower para sistemas de recomendação
- Predição de interações entre usuários e itens
- Experimentação e documentação técnica com Jupyter Notebook
- Uso de Git e GitHub para versionamento de projetos

## Limitações atuais

Este projeto é um protótipo acadêmico voltado ao estudo e à experimentação da arquitetura Two-Tower.

Antes de utilizar uma solução semelhante em produção, seria necessário implementar ou validar:

- Pipeline de dados documentado e reproduzível
- Separação dos dados em treino, validação e teste
- Métricas de avaliação adequadas
- Estratégia de amostragem de interações negativas
- Ajuste de hiperparâmetros
- Comparação com modelos de recomendação de referência
- Pipeline de recuperação e ranqueamento de itens
- API ou serviço em lote para disponibilização de recomendações
- Monitoramento de qualidade, drift de dados e drift de modelo
- Aspectos de privacidade, segurança e vieses do modelo

## Próximas melhorias

- Criar um arquivo `requirements.txt` com versões fixas das dependências
- Documentar a origem, estrutura e preparação do conjunto de dados
- Adicionar análise exploratória dos dados
- Registrar métricas de treinamento, validação e teste
- Comparar o Two-Tower com modelos de baseline
- Implementar amostragem negativa
- Exportar e reutilizar embeddings treinados
- Criar uma API simples para recomendações
- Desenvolver uma interface web para demonstração do modelo

## Autor

**Diego Vitor Lopes Gonçalves Souza**

- Estudante de Engenharia de Software — UNDB
- Especialização em Inteligência Artificial — UNDB
- GitHub: [diegovitor90](https://github.com/diegovitor90)
