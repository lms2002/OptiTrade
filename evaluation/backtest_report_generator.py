import json
import logging
import pandas as pd
from evaluation.metrics.price_metrics import calculate_price_metrics
from evaluation.metrics.direction_metrics import calculate_direction_metrics
from evaluation.metrics.strategy_metrics import calculate_strategy_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_backtest_report(results_df: pd.DataFrame, target_col: str) -> dict:
    """
    Walk-Forward 예측 결과를 바탕으로 3가지 도메인(가격, 방향성, 전략)의 종합 리포트를 생성합니다.
    """
    if results_df.empty or 'predicted' not in results_df.columns:
        logger.error("평가할 예측 결과 데이터가 없습니다.")
        return {}

    logger.info("종합 백테스트 리포트 산출 중...")

    y_true = results_df[target_col].values
    y_pred = results_df['predicted'].values

    # 1. 가격 오차 지표 (연속형 회귀 예측 시)
    price_metrics = calculate_price_metrics(y_true, y_pred)

    # 2. 방향성 예측 변환 (수익률 혹은 가격 변화가 0보다 크면 1(상승), 아니면 0(하락/횡보))
    y_true_dir = (y_true > 0).astype(int)
    
    # 예측값이 확률(0~1)이거나 연속형일 경우 이진화, 이미 0/1 분류값이면 그대로 사용
    if len(set(y_pred)) > 2: 
        y_pred_dir = (y_pred > 0).astype(int)
    else: 
        y_pred_dir = y_pred
        
    direction_metrics = calculate_direction_metrics(y_true_dir, y_pred_dir)

    # 3. 전략 수익성 (간단한 Long Only 전략 가정: 모델이 상승(1)을 예측한 날에만 진입하여 실제 수익률 획득)
    # 실제 타겟이 수익률(Return)로 계산되어 있다고 가정
    portfolio_returns = [
        true_ret if pred == 1 else 0.0 
        for true_ret, pred in zip(y_true, y_pred_dir)
    ]
    strategy_metrics = calculate_strategy_metrics(portfolio_returns)

    # 최종 리포트 취합
    report = {
        "Price_Metrics": price_metrics,
        "Direction_Metrics": direction_metrics,
        "Strategy_Metrics": strategy_metrics
    }

    logger.info("============ [Backtest Evaluation Report] ============")
    print(json.dumps(report, indent=4))
    logger.info("======================================================")
    
    return report