"""
Funções utilitárias para o projeto de previsão de demanda
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """
    Carrega dados de um arquivo CSV
    
    Parameters
    ----------
    filepath : str
        Caminho para o arquivo CSV
        
    Returns
    -------
    pandas.DataFrame
        DataFrame com os dados carregados
    """
    return pd.read_csv(filepath)

def plot_time_series(data, date_column, value_column, title=None):
    """
    Plota uma série temporal
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame com os dados
    date_column : str
        Nome da coluna de data
    value_column : str
        Nome da coluna de valores
    title : str, optional
        Título do gráfico
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data[date_column], data[value_column])
    plt.title(title or f'Série Temporal - {value_column}')
    plt.xlabel('Data')
    plt.ylabel(value_column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt 