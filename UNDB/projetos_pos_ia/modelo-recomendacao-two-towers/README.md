# Modelo de Recomendacao Two-Tower

Modelo de recomendacao de livros usando uma arquitetura Two-Tower com TensorFlow.

## Instalacao

```powershell
python -m pip install -r requirements.txt
```

## Execucao

Na pasta do projeto, execute:

```powershell
python modelo-recomendacao-two-towers.py
```

O script carrega o dataset em `../content/Dataset_avaliacao.csv`, treina o modelo e exibe exemplos de probabilidades de `Like`.
