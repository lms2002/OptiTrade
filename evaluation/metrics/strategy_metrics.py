import numpy as np
import pandas as pd

def calculate_strategy_metrics(portfolio_returns, risk_free_rate=0.02) -> dict:
    """
    일별 포트폴리오 수익률 리스트를 바탕으로 샤프 지수, MDD, 승률을 계산합니다.
    """
    if not portfolio_returns or len(portfolio_returns) == 0:
        return {"Sharpe_Ratio": 0.0, "MDD": 0.0, "Win_Rate": 0.0}

    returns_arr = np.array(portfolio_returns)
    
    # 1. 승률 (수익이 0보다 큰 날의 비율)
    win_rate = np.sum(returns_arr > 0) / len(returns_arr) * 100

    # 2. 연율화 샤프 지수 (Sharpe Ratio, 1년=252 거래일 가정)
    daily_rf = risk_free_rate / 252
    excess_returns = returns_arr - daily_rf
    if np.std(excess_returns) == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    # 3. 최대 낙폭 (MDD: Maximum Drawdown)
    cumulative_returns = (1 + pd.Series(returns_arr)).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    mdd = drawdown.min() * 100

    return {
        "Sharpe_Ratio": round(sharpe_ratio, 4),
        "MDD": round(mdd, 4),
        "Win_Rate": round(win_rate, 4)
    }