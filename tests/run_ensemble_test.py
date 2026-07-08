import os
import sys
import pandas as pd
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ensemble.stacking_ensemble import StackingEnsemble
from evaluation.walk_forward_validator import WalkForwardValidator
from evaluation.backtest_report_generator import generate_backtest_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_test():
    file_path = "feature_engineering/output/final_features.parquet"
    if not os.path.exists(file_path):
        return

    df = pd.read_parquet(file_path)
    df['target_return'] = df['close'].pct_change().shift(-1)
    df['target_direction'] = (df['target_return'] > 0).astype(int)
    df = df.dropna(subset=['target_return'])

    feature_cols = [
        'close', 'volume', 'sma_20', 'sma_50', 'rsi_14', 'bbl_20_2.0', 'bbu_20_2.0',
        'support_20d', 'resistance_20d', 'psr', 'sentiment_score', 'sentiment_ma_5', 'sentiment_delta_5d'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]

    # Stacking Ensemble 모델 적용!
    model = StackingEnsemble()
    validator = WalkForwardValidator(train_window_size=252, test_window_size=21)

    # 전진 검증 수행 (3가지 모델을 묶어서 돌리므로 시간이 약간 더 걸림)
    results_df = validator.validate(df, model, feature_cols, target_col='target_direction')

    if not results_df.empty:
        results_df['target_return'] = df.loc[results_df.index, 'target_return']
        generate_backtest_report(results_df, target_col='target_return')

if __name__ == "__main__":
    run_test()