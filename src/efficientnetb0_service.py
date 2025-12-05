from ray import serve
from fastapi import HTTPException
import torch
from torchvision import models, transforms
from PIL import Image
import io
from typing import Dict, Any, List
from weights_manager import WeightsManager

# Available classification tasks
AVAILABLE_TASKS = ["body_volume", "feet", "gender", "glasses", "hairstyle"]

@serve.deployment(
    name="efficientnet_b0_classifier"
)
class EfficientNetB0Classifier:
    def __init__(self):
        print("🔹 Loading EfficientNetB0 models from Hugging Face...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🌟 PyTorch Device: {self.device}")

        self.models = {}
        self.labels = {}
        self.weights_manager = WeightsManager()
        
        # Load all models from Hugging Face
        self._load_all_models()
        
        # Preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_all_models(self):
        """Load all EfficientNetB0 models from Hugging Face"""
        try:
            # Get available weights from Hugging Face
            weights_info = self.weights_manager.get_available_weights()
            efficientnet_b0_tasks = weights_info.get("efficientnet_b0", [])
            
            if not efficientnet_b0_tasks:
                print("⚠️  No EfficientNetB0 weights found on Hugging Face")
                return
            
            print(f"🔍 Found EfficientNetB0 tasks on Hugging Face: {efficientnet_b0_tasks}")
            
            # Load từng model
            for task_name in efficientnet_b0_tasks:
                try:
                    print(f"📥 Loading EfficientNetB0 model for task: {task_name}")
                    num_classes = len(self._generate_labels_for_task(task_name))
                    model = models.efficientnet_b0(weights=None, num_classes=num_classes)
            
                    state_dict = self.weights_manager.load_model_state_dict("efficientnet_b0", task_name)
                    if state_dict is None:
                        print(f"❌ Failed to load weights for task '{task_name}'")
                        continue
            
                    model.load_state_dict(state_dict, strict=False)
                    model.eval()

                    model = model.to(self.device)

                    self.models[task_name] = model
                    self.labels[task_name] = self._generate_labels_for_task(task_name)
                    print(f"✅ Loaded EfficientNetB0 model for task: {task_name}")
            
                except Exception as e:
                    print(f"❌ Error loading model for task '{task_name}': {e}")

            
            print(f"📊 Loaded {len(self.models)} EfficientNetB0 models: {list(self.models.keys())}")
            
        except Exception as e:
            print(f"❌ Error in _load_all_models: {e}")
            print("💡 Falling back to available tasks list")
            # Fallback - try to load task manually
            for task_name in AVAILABLE_TASKS:
                try:
                    state_dict = self.weights_manager.load_model_state_dict("efficientnet_b0", task_name)
                    if state_dict:
                        model = models.efficientnet_b0(weights=None)
                        model.load_state_dict(state_dict, strict=False)
                        model.eval()
                        self.models[task_name] = model
                        self.labels[task_name] = self._generate_labels_for_task(task_name)
                        print(f"✅ Loaded EfficientNetB0 model for task: {task_name}")
                except Exception as e:
                    print(f"❌ Failed to load task '{task_name}': {e}")

    def _generate_labels_for_task(self, task_name: str) -> List[str]:
        """Generate labels cho từng task"""
        if task_name == "gender":
            return ["male", "female"]
        elif task_name == "glasses":
            return ["yes", "sunglass", "no"]
        elif task_name == "body_volume":
            return ["thin", "medium", "fat", "unknown"]
        elif task_name == "feet":
            return ["sport", "classic", "high heels", "boots", "sandals", "nothing"]
        elif task_name == "hairstyle":
            return ["bald", "short", "medium", "long", "horse tail"]
        else:
            # Default labels if task is unknown
            return [f"class_{i}" for i in range(2)]  # Default 2 classes

    async def predict(self, image_bytes: bytes, task: str = "gender") -> Dict[str, Any]:
        """Classify image using EfficientNetB0 for specific task"""
        try:
            if task not in self.models:
                raise HTTPException(status_code=400, detail=f"Task '{task}' not available. Available tasks: {list(self.models.keys())}")
            
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            input_tensor = self.preprocess(image).unsqueeze(0)

            input_tensor = input_tensor.to(self.device)

            # Get model for task
            model = self.models[task]
            labels = self.labels[task]

            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                # Move results to CPU before processing if needed (ensure float compatibility)
                if self.device.type == 'cuda':
                    probabilities = probabilities.to("cpu")
                
                # Get predictions for all classes
                results = []
                for i, prob in enumerate(probabilities):
                    if i < len(labels):
                        results.append({
                            "label": labels[i],
                            "confidence": float(prob.item())
                        })

            return {
                "model": "efficientnet_b0",
                "task": task,
                "predictions": results
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

        # Keep __call__ if a default endpoint is needed, but it is no longer predict
    async def __call__(self, *args, **kwargs):
        # If Router calls handle.remote() instead of handle.predict.remote(),
        # Ray will call this function. We can redirect it to predict.
        if len(args) >= 1 and isinstance(args[0], bytes):
            return await self.predict(*args, **kwargs)
        
        # This function should not be called directly via HTTP as Router has explicit routing.
        return {"error": "Use the /predict endpoint or call the predict method."}

    async def get_available_tasks(self) -> List[str]:
        """Get available tasks"""
        return list(self.models.keys())

def app(config: dict | None = None):
    # Expose only the internal classifier deployment (no HTTP routes)
    return EfficientNetB0Classifier.bind()