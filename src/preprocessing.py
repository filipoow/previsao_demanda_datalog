# src/preprocessing.py
import pandas as pd
import numpy as np

def gerar_demanda_diaria(path_raw_pedidos: str, path_output: str) -> pd.DataFrame:
    """
    Carrega dados de pedidos, filtra entregues, agrega quantidade por dia e SKU
    e salva em um CSV de saída.

    :param path_raw_pedidos: caminho para o CSV original de pedidos
    :param path_output: caminho onde salvar demanda_diaria.csv
    :return: DataFrame com demanda diária (date, sku, total_quantity)
    """
    pedidos = pd.read_csv(path_raw_pedidos, parse_dates=["date"])
    # Filtrar apenas entregues
    pedidos = pedidos[pedidos["order_status"] == "Delivered"].copy()
    pedidos["date"] = pedidos["date"].dt.date  # extrair apenas a data
    
    demanda_diaria = (
        pedidos
        .groupby(["date", "sku"], as_index=False)
        .agg(total_quantity=("quantity", "sum"))
        .sort_values(["sku", "date"])
    )
    
    demanda_diaria.to_csv(path_output, index=False)
    return demanda_diaria

def clean_data(df):
    """
    Realiza limpeza básica dos dados
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame com os dados brutos
        
    Returns
    -------
    pandas.DataFrame
        DataFrame com os dados limpos
    """
    # Remove linhas com valores nulos
    df = df.dropna()
    
    # Remove duplicatas
    df = df.drop_duplicates()
    
    return df

def aggregate_daily_demand(df, date_column, value_column):
    """
    Agrega dados para demanda diária
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame com os dados
    date_column : str
        Nome da coluna de data
    value_column : str
        Nome da coluna de valores
        
    Returns
    -------
    pandas.DataFrame
        DataFrame com demanda diária agregada
    """
    # Converte coluna de data para datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Agrega por data
    daily_demand = df.groupby(date_column)[value_column].sum().reset_index()
    
    return daily_demand

def resample_to_daily(df, date_column, value_column, fill_method='ffill'):
    """
    Resample os dados para frequência diária, preenchendo valores faltantes
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame com os dados
    date_column : str
        Nome da coluna de data
    value_column : str
        Nome da coluna de valores
    fill_method : str, optional
        Método para preencher valores faltantes ('ffill', 'bfill', 'zero', etc)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame com dados resampled para frequência diária
    """
    # Garante que a coluna de data está no formato datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Define o range de datas
    date_range = pd.date_range(
        start=df[date_column].min(),
        end=df[date_column].max(),
        freq='D'
    )
    
    # Cria um DataFrame com todas as datas
    df_daily = pd.DataFrame({date_column: date_range})
    
    # Merge com os dados originais
    df_daily = df_daily.merge(
        df[[date_column, value_column]],
        on=date_column,
        how='left'
    )
    
    # Preenche valores faltantes
    if fill_method == 'ffill':
        df_daily[value_column] = df_daily[value_column].fillna(method='ffill')
    elif fill_method == 'bfill':
        df_daily[value_column] = df_daily[value_column].fillna(method='bfill')
    elif fill_method == 'zero':
        df_daily[value_column] = df_daily[value_column].fillna(0)
    else:
        raise ValueError(f"Método de preenchimento '{fill_method}' não reconhecido")
    
    return df_daily
