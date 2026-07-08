import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistryClient:
    def __init__(self):
        self.tracking_uri = "http://localhost:5000"
        mlflow.set_tracking_uri(self.tracking_uri)

    def register_model(self, model, model_name: str, run_name: str = "Register_Model_Run"):
        """
        학습이 완료된 모델 객체를 MLflow Model Registry에 등록합니다.
        """
        if mlflow.active_run():
            mlflow.end_run()
            
        with mlflow.start_run(run_name=run_name) as run:
            # sklearn 래퍼를 사용하여 모델 로깅 및 레지스트리 등록
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )
            logger.info(f"모델 '{model_name}'이(가) MLflow Registry에 성공적으로 등록되었습니다.")
            logger.info(f"Run ID: {run.info.run_id}")