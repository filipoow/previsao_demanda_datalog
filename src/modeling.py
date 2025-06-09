# src/modeling.py
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def treinar_sarima(data: pd.DataFrame, order: tuple, seasonal_order: tuple, exog_cols: list, n_test_days: int = 90):
    """
    Treina um modelo SARIMAX sobre a série 'data', que deve ser um DataFrame com índice datetime
    e colunas ['y'] + exog_cols. Retorna o objeto fitted_model, RMSE no conjunto de teste e séries real/predita.
    """
    df = data.copy()
    train_end = df.index.max() - pd.Timedelta(days=n_test_days)
    train = df.loc[:train_end]
    test = df.loc[train_end + pd.Timedelta(days=1):]

    model = SARIMAX(
        train["y"], 
        order=order,
        seasonal_order=seasonal_order,
        exog=train[exog_cols],
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted_model = model.fit(disp=False)

    predictions = fitted_model.predict(
        start=test.index[0],
        end=test.index[-1],
        exog=test[exog_cols]
    )

    rmse = np.sqrt(mean_squared_error(test["y"], predictions))
    return fitted_model, rmse, test["y"], predictions

def treinar_random_forest(data: pd.DataFrame, feature_cols: list, target_col: str = "y", n_test_days: int = 90):
    """
    Treina RandomForestRegressor sobre 'data', com features e target especificados.
    Retorna o modelo treinado, MAE no teste, séries real e predita.
    """
    df = data.copy()
    train_end = df.index.max() - pd.Timedelta(days=n_test_days)
    train = df.loc[:train_end]
    test = df.loc[train_end + pd.Timedelta(days=1):]

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return rf, mae, y_test, y_pred
