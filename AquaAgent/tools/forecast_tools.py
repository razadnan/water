"""
Forecasting tool for municipal water demand using XGBoost and Rolling Baselines.
Evaluates forecast accuracy with Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE)
against a naive "same-hour-last-week" baseline.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from config import settings


class WaterDemandForecaster:
    def __init__(self, zone: str):
        self.zone = zone
        self.model = None
        self.features = ["hour", "dayofweek", "lag_24h", "lag_168h", "rolling_mean_24h"]

    def _prepare_features(self, df_zone: pd.DataFrame) -> pd.DataFrame:
        df = df_zone.copy().sort_values("timestamp").reset_index(drop=True)
        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        df["lag_24h"] = df["consumption_l"].shift(24)
        df["lag_168h"] = df["consumption_l"].shift(168)  # same hour last week
        df["rolling_mean_24h"] = df["consumption_l"].shift(1).rolling(24, min_periods=6).mean()
        return df.dropna().reset_index(drop=True)

    def train_and_evaluate(self, test_days: int = 7) -> dict:
        """
        Trains an XGBoost regression model on historical days,
        evaluates performance on the hold-out test set, and benchmarks
        against a naive persistent baseline.
        """
        import xgboost as xgb
        from tools import water_tools as wt

        df = wt.load_data()
        df_z = df[df["zone"] == self.zone].copy()
        df_z["timestamp"] = pd.to_datetime(df_z["timestamp"])

        feat_df = self._prepare_features(df_z)
        if len(feat_df) < 168:
            return {"error": f"Insufficient historical data to train forecaster ({len(feat_df)} rows)."}

        # Hold out the last test_days * 24 hours
        cutoff = feat_df["timestamp"].max() - pd.Timedelta(days=int(test_days))
        train_df = feat_df[feat_df["timestamp"] <= cutoff]
        test_df = feat_df[feat_df["timestamp"] > cutoff]

        if len(test_df) == 0:
            return {"error": "Test set empty. Please expand historical date range."}

        X_train, y_train = train_df[self.features], train_df["consumption_l"]
        X_test, y_test = test_df[self.features], test_df["consumption_l"]

        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        # Predictions
        preds = self.model.predict(X_test)
        preds = np.maximum(preds, 0.0)

        # Naive Baseline: same hour last week (lag_168h)
        naive_preds = test_df["lag_168h"].values

        # Calculate metrics
        xgb_mae = float(np.mean(np.abs(preds - y_test.values)))
        xgb_mape = float(np.mean(np.abs((preds - y_test.values) / np.maximum(y_test.values, 1.0))) * 100.0)

        naive_mae = float(np.mean(np.abs(naive_preds - y_test.values)))
        naive_mape = float(np.mean(np.abs((naive_preds - y_test.values) / np.maximum(y_test.values, 1.0))) * 100.0)

        improvement_pct = round((naive_mae - xgb_mae) / naive_mae * 100.0, 2)

        return {
            "zone": self.zone,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "xgboost_mae": round(xgb_mae, 1),
            "xgboost_mape_pct": round(xgb_mape, 2),
            "naive_baseline_mae": round(naive_mae, 1),
            "naive_baseline_mape_pct": round(naive_mape, 2),
            "improvement_over_naive_pct": improvement_pct,
            "latest_predicted_daily_l": round(float(np.sum(preds[-24:])), 1),
            "model_type": "XGBoost Regressor (lagged + rolling features)"
        }

    def forecast_next_24h(self) -> dict:
        """
        Forecasts expected consumption for the next 24-hour cycle.
        """
        res = self.train_and_evaluate(test_days=7)
        return {
            "zone": self.zone,
            "expected_next_24h_l": res.get("latest_predicted_daily_l"),
            "model_confidence_mape": res.get("xgboost_mape_pct"),
            "baseline_improvement": res.get("improvement_over_naive_pct")
        }


def forecast_demand(zone: str) -> dict:
    """Tool wrapper exposed to agent registry."""
    forecaster = WaterDemandForecaster(zone)
    return forecaster.train_and_evaluate(test_days=7)
