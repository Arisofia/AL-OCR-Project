from label_studio_sdk import Client
from src.strategies.hybrid import HybridSampler
from src.core.model_interface import OCRBaseModel
import os

class ActiveLearningOrchestrator:
    def __init__(self, model: OCRBaseModel, ls_url: str, ls_key: str):
        self.model = model
        self.sampler = HybridSampler()
        self.ls_client = Client(url=ls_url, api_key=ls_key)
        self.project = self.ls_client.get_project(1) # Assuming project ID 1

    def run_cycle(self, unlabeled_data_path: str):
        print("--- Starting AL Cycle ---")
        images, filenames = self.load_images(unlabeled_data_path)
        print("Scoring images...")
        selected_indices = self.sampler.select_batch(self.model, images, n_samples=50)
        tasks = []
        for idx in selected_indices:
            prediction = self.model.predict(images[idx])
            tasks.append({
                "data": {"image": filenames[idx]},
                "predictions": [{
                    "model_version": "v1.0",
                    "result": [{
                        "from_name": "label",
                        "to_name": "image",
                        "type": "textarea",
                        "value": {"text": [prediction['text']]}
                    }]
                }]
            })
        self.project.import_tasks(tasks)
        print(f"Uploaded {len(tasks)} tasks to Label Studio for human review.")

    def load_images(self, path):
        # ... logic to load images ...
        return [], []
