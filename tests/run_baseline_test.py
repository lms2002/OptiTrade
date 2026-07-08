import os
import sys
import pandas as pd
import logging

# 프로젝트 최상위 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline.xgboost_model import XGBoostBaseline
from evaluation.walk_forward_validator import WalkForwardValidator
from evaluation.backtest_report_generator import generate_backtest_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_test():
    file_path = "feature_engineering/output/final_features.parquet"
    if not os.path.exists(file_path):
        logger.error("Feature Set 파일이 없습니다. feature_merger.py를 먼저 실행하세요.")
        return

    df = pd.read_parquet(file_path)

    # 1. 정답(Target) 변수 생성: 미래 참조가 아니도록 '다음 날'의 수익률을 현재 행의 타겟으로 설정
    df['target_return'] = df['close'].pct_change().shift(-1)
    
    # 방향성 분류 타겟: 다음날 수익률이 0보다 크면 1(상승), 아니면 0(하락)
    df['target_direction'] = (df['target_return'] > 0).astype(int)
    
    # shift로 인해 발생한 맨 마지막 날 결측치 제거
    df = df.dropna(subset=['target_return'])

    # 2. 학습에 사용할 입력 피처(Feature) 리스트
    feature_cols = [
        'close', 'volume', 'sma_20', 'sma_50', 'rsi_14', 'bbl_20_2.0', 'bbu_20_2.0',
        'support_20d', 'resistance_20d', 'psr', 'sentiment_score', 'sentiment_ma_5', 'sentiment_delta_5d'
    ]
    # 실제 존재하는 컬럼만 필터링
    feature_cols = [col for col in feature_cols if col in df.columns]

    # 3. 모델 및 Walk-Forward 검증기 세팅
    # 252일(약 1년) 학습하여 21일(약 1개월) 예측하며 슬라이딩
    model = XGBoostBaseline(task_type='classification')
    validator = WalkForwardValidator(train_window_size=252, test_window_size=21)

    # 4. 전진 검증 수행
    # 분류 모델이므로 타겟은 'target_direction'을 줍니다.
    results_df = validator.validate(df, model, feature_cols, target_col='target_direction')

    # 5. 리포트 생성을 위한 보정 (전략 수익률 계산을 위해 타겟 수익률 컬럼 복원)
    if not results_df.empty:
        # results_df의 인덱스를 원본 df와 매핑하여 실제 수익률(target_return)을 가져옵니다.
        results_df['target_return'] = df.loc[results_df.index, 'target_return']
        
        # 실제 수익률 기준으로 리포트 생성
        generate_backtest_report(results_df, target_col='target_return')

if __name__ == "__main__":
    run_test()