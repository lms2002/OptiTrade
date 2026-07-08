import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowTracker:
    def __init__(self, experiment_name="Stock_Predictor_DualTrack"):
        """
        로컬 Docker 컨테이너에서 구동 중인 MLflow 서버에 연결합니다.
        """
        self.tracking_uri = "http://localhost:5000"
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow 서버({self.tracking_uri}) 연결 및 실험 '{experiment_name}' 세팅 완료")

    def log_experiment(self, run_name: str, params: dict, metrics: dict):
        """
        파라미터와 평가 지표를 MLflow에 기록합니다.
        """
        # 현재 활성화된 Run이 있으면 종료 (충돌 방지)
        if mlflow.active_run():
            mlflow.end_run()
            
        with mlflow.start_run(run_name=run_name):
            # 파라미터 로깅
            if params:
                mlflow.log_params(params)
            
            # 평가 지표 로깅 (중첩 딕셔너리 평탄화)
            flat_metrics = {}
            for category, metric_data in metrics.items():
                for key, value in metric_data.items():
                    flat_metrics[f"{category}_{key}"] = value
                    
            mlflow.log_metrics(flat_metrics)
            logger.info(f"실험 '{run_name}' 메타데이터 MLflow 기록 완료!")