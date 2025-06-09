# Previsão de Demanda

Este projeto implementa um sistema de previsão de demanda utilizando técnicas de machine learning e análise de séries temporais.

## Estrutura do Projeto

```
previsao_demanda/
│
├── data/                    
│   ├── raw/                 # Dados brutos (CSV gerados ou coletados)
│   └── processed/           # Dados tratados/prontos para modelagem
│
├── notebooks/               # Jupyter notebooks para ETL e experimentação
│
├── src/                     # Código-fonte modularizado
│   ├── utils.py            # Funções auxiliares
│   ├── preprocessing.py    # Scripts para limpeza e agregação
│   └── modeling.py         # Funções para treinar e avaliar modelos
│
└── requirements.txt         # Dependências do projeto
```

## Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd previsao_demanda
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

1. Coloque seus dados brutos na pasta `data/raw/`
2. Execute os notebooks na pasta `notebooks/` em ordem:
   - `01_coleta_e_preprocessamento.ipynb`
   - `02_analise_exploratoria.ipynb`
   - `03_modelagem_previsao.ipynb`

## Funcionalidades

- Carregamento e limpeza de dados
- Análise exploratória de dados
- Pré-processamento e feature engineering
- Treinamento de modelos de previsão
- Avaliação de modelos
- Visualização de resultados

## Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request 