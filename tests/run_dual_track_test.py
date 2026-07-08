import os
import sys
import pandas as pd
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ensemble.dual_track_model import DualTrackModel
from evaluation.backtest_report_generator import generate_backtest_report
from mlops.mlflow_tracker import MLflowTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_dual_track_test():
    file_path = "feature_engineering/output/final_features.parquet"
    if not os.path.exists(file_path):
        logger.error("Feature Set 파일이 없습니다.")
        return

    df = pd.read_parquet(file_path)
    
    # 이중 트랙용 정답지 생성
    df['target_return'] = df['close'].pct_change().shift(-1)
    df['target_direction'] = (df['target_return'] > 0).astype(int)
    df = df.dropna(subset=['target_return'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    feature_cols = [
        'volume', 'sma_20', 'sma_50', 'rsi_14', 'bbl_20_2.0', 'bbu_20_2.0',
        'support_20d', 'resistance_20d', 'psr', 'sentiment_score', 'sentiment_ma_5', 
        'sentiment_delta_5d', 'treasury_10y', 'insider_mspr', 'insider_net_change'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]

    train_window_size = 252
    test_window_size = 21
    total_steps = (len(df) - train_window_size) // test_window_size
    
    # 0.05% 이상의 기대수익이 있을 때만 진입
    model = DualTrackModel(ev_threshold=0.0005)
    results = []

    logger.info(f"Dual Track Walk-Forward 검증 시작 (총 {total_steps} 스텝)...")

    for step in range(total_steps):
        train_start = step * test_window_size
        train_end = train_start + train_window_size
        test_end = train_end + test_window_size

        if test_end > len(df):
            break

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[train_end:test_end]

        X_train = train_df[feature_cols]
        y_train_dir = train_df['target_direction']
        y_train_ret = train_df['target_return']
        
        X_test = test_df[feature_cols]

        # 모델 훈련 및 예측
        model.fit(X_train, y_train_dir, y_train_ret)
        preds = model.predict(X_test)

        step_result = test_df[['date', 'ticker', 'close', 'target_return']].copy()
        step_result['predicted'] = preds
        results.append(step_result)

    if not results:
        logger.error("검증 결과가 없습니다.")
        return

    final_results = pd.concat(results, ignore_index=True)
    logger.info(f"검증 완료! 총 {len(final_results)}일의 실거래 모의 평가 진행")

    # 1. 결과 리포트 산출
    report = generate_backtest_report(final_results, target_col='target_return')

    # 2. MLflow 기록 로직
    tracker = MLflowTracker()
    params = {
        "model_type": "DualTrack_LGBM_XGB",
        "ev_threshold": 0.0005,
        "train_window": train_window_size,
        "test_window": test_window_size
    }
    
    if report:
        tracker.log_experiment(run_name="Dual_Track_V1", params=params, metrics=report)

if __name__ == "__main__":
    run_dual_track_test()