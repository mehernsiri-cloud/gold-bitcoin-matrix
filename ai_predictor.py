# ai_predictor.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

def prepare_data(df, target_col="actual"):
    """Prepare historical data for ML model"""
    df = df.dropna(subset=[target_col])
    df = df.sort_values("timestamp")
    df['timestamp_ordinal'] = df['timestamp'].apply(lambda x: x.toordinal())
    X = df[['timestamp_ordinal']].values
    y = df[target_col].values
    return X, y

def predict_next_n(df_actual, df_pred, asset_name="Gold", n_steps=5):
    """Predict next n_steps using past actual + predicted prices"""
    # Merge actual + predicted
    df = pd.merge_asof(
        df_pred[df_pred['asset']==asset_name].sort_values('timestamp'),
        df_actual[['timestamp', f'{asset_name.lower()}_actual']].rename(columns={f'{asset_name.lower()}_actual':'actual'}),
        on='timestamp', direction='backward'
    )
    
    df['target'] = df['actual'].combine_first(df['predicted_price'])
    df = df.dropna(subset=['target'])
    
    if df.empty:
        return pd.DataFrame(columns=['timestamp', 'predicted_price'])
    
    X, y = prepare_data(df, 'target')
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future dates (daily)
    last_date = df['timestamp'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, n_steps+1)]
    X_future = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    y_pred = model.predict(X_future)
    
    df_future = pd.DataFrame({
        'timestamp': future_dates,
        'predicted_price': y_pred
    })
    return df_future
