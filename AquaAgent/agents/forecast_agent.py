from agents.base_agent import BaseAgent


class ForecastAgent(BaseAgent):
    name = "Demand Forecasting Agent"
    role = "Forecast 24-hour water demand using XGBoost machine learning and evaluate forecast variance."
    tool_names = ["forecast_demand"]
    required_tools = ["forecast_demand"]

    def offline_summary(self, obs, context):
        f = obs.get("forecast_demand", {})
        if "error" in f:
            return f"Forecasting unavailable for {context['zone']}: {f['error']}"
        mape = f.get("xgboost_mape_pct", 0.0)
        mae = f.get("xgboost_mae", 0.0)
        impr = f.get("improvement_over_naive_pct", 0.0)
        return (f"Demand Forecast for {context['zone']} (XGBoost ML): MAE {mae:.1f} L, "
                f"MAPE {mape:.2f}%. Model achieved a {impr:+.1f}% accuracy gain over the naive weekly baseline.")
