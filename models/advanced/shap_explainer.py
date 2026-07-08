import os
import sys
import pandas as pd
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExplainer:
    def __init__(self):
        """
        TreeExplainer를 활용하기 위해 이중 트랙의 핵심인 LightGBM 분류기를 사용합니다.
        Phase 5에서 찾은 최적 파라미터를 동일하게 적용합니다.
        """
        self.model = lgb.LGBMClassifier(
            n_estimators=112, max_depth=5, learning_rate=0.079, 
            subsample=0.588, colsample_bytree=0.878, random_state=42, verbose=-1
        )
        self.feature_cols = [
            'volume', 'sma_20', 'sma_50', 'rsi_14', 'bbl_20_2.0', 'bbu_20_2.0',
            'support_20d', 'resistance_20d', 'psr', 'sentiment_score', 'sentiment_ma_5', 
            'sentiment_delta_5d', 'treasury_10y', 'insider_mspr', 'insider_net_change'
        ]

    def explain(self, data_path="feature_engineering/output/final_features.parquet"):
        if not os.path.exists(data_path):
            logger.error("데이터 파일이 존재하지 않습니다.")
            return

        df = pd.read_parquet(data_path)
        
        # 타겟 변수 생성
        df['target_return'] = df['close'].pct_change().shift(-1)
        df['target_direction'] = (df['target_return'] > 0).astype(int)
        df = df.dropna(subset=['target_return'])

        # ========== 수정된 부분: 실제로 데이터프레임에 존재하는 컬럼만 필터링 ==========
        actual_cols = [col for col in self.feature_cols if col in df.columns]
        X = df[actual_cols]
        y = df['target_direction']
        # =================================================================================

        # 모델 학습 (전체 데이터로 피처 중요도 파악)
        logger.info("SHAP 분석용 모델 학습 중...")
        self.model.fit(X, y)

        # SHAP Explainer 생성 및 값 계산
        logger.info("SHAP Value 계산 중... (시간이 조금 걸릴 수 있습니다)")
        explainer = shap.TreeExplainer(self.model)
        
        # SHAP 값 추출
        shap_values = explainer.shap_values(X)
        
        # LightGBM 이진 분류의 경우 리스트 형태로 반환될 수 있으므로 양성 클래스(1) 값만 추출
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # 결과 시각화 및 저장
        output_dir = "models/advanced/output"
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, show=False)
        save_path = f"{output_dir}/shap_summary.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"🎉 SHAP 시각화 완료! 어떤 피처가 가장 큰 영향을 미쳤는지 확인해 보세요: {save_path}")

if __name__ == "__main__":
    explainer = ModelExplainer()
    explainer.explain()