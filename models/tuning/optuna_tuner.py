import os
import sys
import optuna
import pandas as pd
import lightgbm as lgb
import logging
from sklearn.metrics import accuracy_score

# 프로젝트 최상위 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation.walk_forward_validator import WalkForwardValidator

# Optuna 자체의 불필요한 로그는 숨기고 결과만 집중해서 보기 위함
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """데이터 로드 및 타겟 생성 (매 Trial마다 반복하지 않도록 1회만 로드)"""
    file_path = "feature_engineering/output/final_features.parquet"
    df = pd.read_parquet(file_path)
    
    # 다음날 수익률 및 방향성 타겟 생성
    df['target_return'] = df['close'].pct_change().shift(-1)
    df['target_direction'] = (df['target_return'] > 0).astype(int)
    df = df.dropna(subset=['target_return'])
    
    feature_cols = [
        'close', 'volume', 'sma_20', 'sma_50', 'rsi_14', 'bbl_20_2.0', 'bbu_20_2.0',
        'support_20d', 'resistance_20d', 'psr', 'sentiment_score', 'sentiment_ma_5', 'sentiment_delta_5d'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    return df, feature_cols

def objective(trial, df, feature_cols):
    """Optuna가 최적화할 목적 함수"""
    
    # 과적합을 막기 위한 보수적인 파라미터 탐색 공간 설정
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 30, 150),
        'max_depth': trial.suggest_int('max_depth', 2, 5), # 기존 5에서 2까지 얕게 허용
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9), # 데이터 일부만 사용
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9), # 피처 일부만 사용
        'random_state': 42,
        'verbose': -1
    }

    model = lgb.LGBMClassifier(**params)
    validator = WalkForwardValidator(train_window_size=252, test_window_size=21)

    # 전진 검증 수행
    results_df = validator.validate(df, model, feature_cols, target_col='target_direction')

    if results_df.empty:
        return 0.0
        
    # 최적화 목표: 정확도(Accuracy) 최대화
    acc = accuracy_score(results_df['target_direction'], results_df['predicted'])
    return acc

def run_tuner():
    logger.info("데이터 로드 중...")
    df, feature_cols = load_data()
    
    logger.info("Optuna 튜닝 시작 (총 30회 탐색)... 조금만 기다려주세요!")
    
    # 정확도(Accuracy)를 '최대화(maximize)'하는 방향으로 학습
    study = optuna.create_study(direction="maximize")
    
    # 로컬 환경을 고려하여 n_trials=30 (30번 조합 테스트) 설정
    study.optimize(lambda trial: objective(trial, df, feature_cols), n_trials=30)
    
    logger.info("==================================================")
    logger.info("🎉 튜닝 완료!")
    logger.info(f"최고 정확도 (Best Accuracy): {study.best_value:.4f}")
    logger.info("최적 파라미터 (Best Params):")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")
    logger.info("==================================================")

if __name__ == "__main__":
    run_tuner()