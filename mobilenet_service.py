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
    name="mobilenet_classifier"
)
class MobilenetClassifier:
    def __init__(self):
        print("🔹 Loading MobileNet models from Hugging Face...")
        
        self.models = {}
        self.labels = {}
        self.weights_manager = WeightsManager()
        
        # Load tất cả các MobileNet weights từ Hugging Face
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
        """Load tất cả các MobileNet models từ Hugging Face"""
        try:
            # Get available weights từ Hugging Face
            weights_info = self.weights_manager.get_available_weights()
            mobilenet_tasks = weights_info.get("mobilenet", [])
            
            if not mobilenet_tasks:
                print("⚠️  No MobileNet weights found on Hugging Face")
                return
            
            print(f"🔍 Found MobileNet tasks on Hugging Face: {mobilenet_tasks}")
            
            # Load từng model
            for task_name in mobilenet_tasks:
                try:
                    print(f"📥 Loading MobileNetV2 model for task: {task_name}")
                    num_classes = len(self._generate_labels_for_task(task_name))
                    model = models.mobilenet_v2(weights=None, num_classes=num_classes)
            
                    state_dict = self.weights_manager.load_model_state_dict("mobilenet", task_name)
                    if state_dict is None:
                        print(f"❌ Failed to load weights for task '{task_name}'")
                        continue
            
                    model.load_state_dict(state_dict, strict=False)
                    model.eval()
                    self.models[task_name] = model
                    self.labels[task_name] = self._generate_labels_for_task(task_name)
                    print(f"✅ Loaded MobileNetV2 model for task: {task_name}")
            
                except Exception as e:
                    print(f"❌ Error loading model for task '{task_name}': {e}")

            
            print(f"📊 Loaded {len(self.models)} MobileNet models: {list(self.models.keys())}")
            
        except Exception as e:
            print(f"❌ Error in _load_all_models: {e}")
            print("💡 Falling back to available tasks list")
            # Fallback - try to load từng task manually
            for task_name in AVAILABLE_TASKS:
                try:
                    state_dict = self.weights_manager.load_model_state_dict("mobilenet", task_name)
                    if state_dict:
                        model = models.mobilenet_v2(weights=None)
                        model.load_state_dict(state_dict, strict=False)
                        model.eval()
                        self.models[task_name] = model
                        self.labels[task_name] = self._generate_labels_for_task(task_name)
                        print(f"✅ Loaded MobileNet model for task: {task_name}")
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
            # Default labels nếu không biết task
            return [f"class_{i}" for i in range(2)]  # Default 2 classes

    async def predict(self, image_bytes: bytes, task: str = "gender") -> Dict[str, Any]:
        """Classify image using MobileNet for specific task"""
        try:
            if task not in self.models:
                raise HTTPException(status_code=400, detail=f"Task '{task}' not available. Available tasks: {list(self.models.keys())}")
            
            # Load và preprocess image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            input_tensor = self.preprocess(image).unsqueeze(0)

            # Get model cho task
            model = self.models[task]
            labels = self.labels[task]

            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get predictions for all classes
                results = []
                for i, prob in enumerate(probabilities):
                    if i < len(labels):
                        results.append({
                            "label": labels[i],
                            "confidence": float(prob.item())
                        })

            return {
                "model": "mobilenet",
                "task": task,
                "predictions": results
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

        # Giữ lại __call__ nếu cần endpoint mặc định, nhưng nó không còn là predict nữa
    async def __call__(self, *args, **kwargs):
        # Nếu Router gọi handle.remote() thay vì handle.predict.remote(), 
        # Ray sẽ gọi hàm này. Ta có thể chuyển hướng nó đến predict.
        if len(args) >= 1 and isinstance(args[0], bytes):
            return await self.predict(*args, **kwargs)
        
        # Hàm này không nên được gọi trực tiếp qua HTTP vì Router đã định tuyến rõ ràng.
        return {"error": "Use the /predict endpoint or call the predict method."}

    async def get_available_tasks(self) -> List[str]:
        """Get available tasks"""
        return list(self.models.keys())

def app(config: dict | None = None):
    # Expose only the internal classifier deployment (no HTTP routes)
    return MobilenetClassifier.bind()