import pandas as pd
import numpy as np
import lightgbm as lgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegimeClassifier:
    def __init__(self):
        """
        시장 레짐(상승/하락/횡보)을 분류하는 3-class 분류기입니다.
        거시적인 흐름을 파악하는 것이 목적이므로 트리의 깊이를 얕게 제한합니다.
        """
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,  # 노이즈를 무시하고 큰 줄기만 보도록 제한
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )

    def _create_regime_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        20일 이동평균선의 5일간 변화율(기울기)을 기준으로 레짐을 자동 라벨링합니다.
        2: 상승장 (변화율 > 1%)
        0: 하락장 (변화율 < -1%)
        1: 횡보장 (그 외)
        """
        if 'sma_20' not in df.columns:
            # 기술적 지표가 누락된 경우 종가(close)로 대체
            price_col = 'close'
        else:
            price_col = 'sma_20'
            
        # 5일 전 대비 이평선의 변화율 계산
        slope = df[price_col].pct_change(periods=5).fillna(0)
        
        conditions = [
            (slope > 0.01),
            (slope < -0.01)
        ]
        choices = [2, 0]
        # 조건에 맞지 않으면 기본값 1(횡보) 부여
        regime = np.select(conditions, choices, default=1)
        
        return pd.Series(regime, index=df.index)

    def fit(self, X_train: pd.DataFrame, y_train=None):
        """
        y_train이 주어지지 않더라도, X_train 내부의 가격 데이터를 바탕으로
        스스로 레짐 정답지(Label)를 만들어 학습합니다.
        """
        # 스스로 레짐 라벨 생성
        y_regime = self._create_regime_labels(X_train)
        
        # 모델 훈련
        self.model.fit(X_train, y_regime)
        logger.info("레짐 분류기(Regime Classifier) 학습 완료")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """레짐 클래스(0, 1, 2) 예측 반환"""
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        앙상블 메타 모델에 힌트로 제공하기 위해
        각 레짐(하락/횡보/상승)에 속할 확률 분포를 반환합니다.
        """
        return self.model.predict_proba(X_test)