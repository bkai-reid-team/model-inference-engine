from ray import serve
from fastapi import HTTPException
import torch
from torchvision import models, transforms
from PIL import Image
import io
from typing import Dict, Any, List
from weights_manager import WeightsManager

# Available classification tasks (Cần phải khớp với những gì WeightsManager trả về)
AVAILABLE_TASKS = ["body_volume", "feet", "gender", "glasses", "hairstyle"]

# Tên class deployment
@serve.deployment(
    name="efficientnet_b4_classifier"
)
class EfficientNetB4Classifier:
    def __init__(self):
        print("🔹 Loading EfficientNetB4 models...")
        
        # Xác định thiết bị (GPU hoặc CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚙️ Using device: {self.device}")
        
        self.models = {}
        self.labels = {}
        self.weights_manager = WeightsManager()
        
        # Load tất cả các EfficientNetB4 weights
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
        """Load tất cả các EfficientNetB4 models"""
        try:
            # Get available weights
            weights_info = self.weights_manager.get_available_weights()
            efficientnet_b4_tasks = weights_info.get("efficientnet_b4", [])
            
            if not efficientnet_b4_tasks:
                print("⚠️ No EfficientNetB4 weights found.")
                return
            
            print(f"🔍 Found EfficientNetB4 tasks: {efficientnet_b4_tasks}")
            
            # Load từng model
            for task_name in efficientnet_b4_tasks:
                try:
                    print(f"📥 Loading EfficientNetB4 model for task: {task_name}")
                    
                    # ✅ Xác định số lớp đúng cho từng task
                    num_classes = len(self._generate_labels_for_task(task_name))
                    
                    # ✅ Tạo model với số lớp tương ứng
                    model = models.efficientnet_b4(weights=None, num_classes=num_classes)
                    
                    # Load state dict
                    state_dict = self.weights_manager.load_model_state_dict("efficientnet_b4", task_name)
                    if state_dict is None:
                        print(f"❌ Failed to load weights for task '{task_name}'")
                        continue
            
                    # Load weights vào model
                    model.load_state_dict(state_dict, strict=False)
                    
                    # Chuyển model sang GPU/thiết bị đã chọn
                    model.to(self.device)
                    model.eval()
                    
                    # Lưu model và nhãn
                    self.models[task_name] = model
                    self.labels[task_name] = self._generate_labels_for_task(task_name)
                    
                    print(f"✅ Loaded EfficientNetB4 model for task: {task_name}")
            
                except Exception as e:
                    print(f"❌ Error loading model for task '{task_name}': {e}")
                    raise 
            
            print(f"📊 Loaded {len(self.models)} EfficientNetB4 models: {list(self.models.keys())}")
            
        except Exception as e:
            print(f"❌ Critical Error in _load_all_models: {e}")
            raise 

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

    # FIX: Định nghĩa hàm predict rõ ràng mà Router đang tìm kiếm.
    async def predict(self, image_bytes: bytes, task: str = "gender") -> Dict[str, Any]:
        """Classify image using EfficientNetB4 for specific task"""
        # Toàn bộ logic xử lý ảnh được chuyển từ __call__ sang đây
        try:
            if task not in self.models:
                raise HTTPException(status_code=400, detail=f"Task '{task}' not available. Available tasks: {list(self.models.keys())}")
            
            # Load và preprocess image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Chuyển input_tensor sang thiết bị (GPU)
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

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
                "model": "efficientnet_b4",
                "task": task,
                "predictions": results
            }
            
        except HTTPException:
            raise
        except Exception as e:
            # Nâng lỗi lên HTTP 500 để client thấy
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
    return EfficientNetB4Classifier.bind()