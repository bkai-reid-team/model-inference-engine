from typing import Any, Dict
from fastapi import FastAPI, HTTPException, UploadFile, Request
from pydantic import BaseModel
from ray import serve


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int | None = 256
    temperature: float | None = 0.7


class TextRequest(BaseModel):
    text: str


# Tạo FastAPI app với endpoints đơn giản
app = FastAPI(
    title="Ray Serve Multi-Model API",
    description="Simple endpoints for AI models",
    version="1.0.0"
)


@serve.deployment() 
@serve.ingress(app)
class Router:
    def __init__(
        self,
        qwen_handle = None,
        embedding_handle = None,
        mobilenet_classifier_handle = None,
        efficientnet_b0_classifier_handle = None,
        efficientnet_b4_classifier_handle = None,
    ) -> None:
        # Nhận sẵn deployment handles thông qua DAG binding
        self.qwen = qwen_handle
        self.embed = embedding_handle
        self.mobilenet_classifier = mobilenet_classifier_handle
        self.efficientnet_b0_classifier = efficientnet_b0_classifier_handle
        self.efficientnet_b4_classifier = efficientnet_b4_classifier_handle

    @app.get("/")
    async def root(self) -> Dict[str, Any]:
        return {
            "message": "Ray Serve Multi-Model API",
            "endpoints": {
                "generation": "/generate",
                "embedding": "/embed", 
                "classification": "/classification",
                "mobilenet_v2": "/mobilenet_v2",
                "efficientnet_b0": "/efficientnet_b0",
                "efficientnet_b4": "/efficientnet_b4",
                "predict": "/predict"
            }
        }

    @app.post("/generate")
    async def generate(self, body: GenerateRequest) -> Dict[str, Any]:
        """Text generation endpoint"""
        if not body.prompt:
            raise HTTPException(400, "Missing 'prompt'")
        if self.qwen is None:
            raise HTTPException(503, "Qwen service not available")
            
        # Gọi service với format phù hợp
        result = await self.qwen.generate.remote({
            "prompt": body.prompt,
            "max_tokens": body.max_tokens,
            "temperature": body.temperature
        })
        return result

    @app.post("/embed")
    async def embed(self, body: TextRequest) -> Dict[str, Any]:
        """Text embedding endpoint"""
        if not body.text:
            raise HTTPException(400, "Missing 'text'")
        if self.embed is None:
            raise HTTPException(503, "Embedding service not available")
            
        result = await self.embed.encode.remote({"text": body.text})
        return result

    @app.post("/mobilenet_v2")
    async def image_classification(self, file: UploadFile) -> Dict[str, Any]:
        """Image classification endpoint using MobileNetV2"""
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        # FIX: Dùng self.mobilenet_classifier thay vì self.mobilenet
        if self.mobilenet_classifier is None: 
            raise HTTPException(503, "MobileNet service not available")
    
        image_bytes = await file.read()
    
        # FIX: Dùng self.mobilenet_classifier.predict.remote(file)
        result = await self.mobilenet_classifier.predict.remote(image_bytes) 
        return result

    @app.post("/efficientnet_b0")
    async def efficientnet_b0(self, file: UploadFile) -> Dict[str, Any]:
        """Image classification endpoint using EfficientNetB0"""
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        # Handle đã được đặt tên đúng
        if self.efficientnet_b0_classifier is None: 
            raise HTTPException(503, "EfficientNetB0 service not available")
                
        image_bytes = await file.read()
    
        # FIX: Dùng self.efficientnet_b0_classifier.predict.remote(file)
        result = await self.efficientnet_b0_classifier.predict.remote(image_bytes) 
        return result

    @app.post("/efficientnet_b4")
    async def efficientnet_b4(self, file: UploadFile) -> Dict[str, Any]:
        """Image classification endpoint using EfficientNetB4"""
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        # Handle đã được đặt tên đúng
        if self.efficientnet_b4_classifier is None: 
            raise HTTPException(503, "EfficientNetB4 service not available")

        image_bytes = await file.read()
        
        # FIX: Dùng self.efficientnet_b4_classifier.predict.remote(file)
        result = await self.efficientnet_b4_classifier.predict.remote(image_bytes) 
        return result

    @app.post("/predict")
    async def predict(self, request: Request, file: UploadFile | None = None, model: str | None = None, deployment: str | None = None, task: str | None = None) -> Dict[str, Any]:
        """Unified prediction endpoint routing to text or image models.

        - JSON body with {prompt,...} -> Qwen
        - JSON body with {text} -> Embedding
        - multipart/form-data with image `file` (+ optional `model`, `task`) -> image classifiers
        """
        content_type = request.headers.get("content-type", "")

        # Image branch: multipart form with file
        if file is not None or content_type.startswith("multipart/"):
            if file is None:
                raise HTTPException(400, "Missing file for image prediction")
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(400, "File must be an image")

            # Default image model; prefer explicit deployment selector
            selected_model = (deployment or model or "mobilenet").lower()
            selected_task = task or "gender"

            image_bytes = await file.read()

            if selected_model in ("mobilenet", "mobile_net", "mobile"):
                if self.mobilenet_classifier is None:
                    raise HTTPException(503, "MobileNet classifier not available")
                # FIX: Sửa thành .predict.remote() để đồng nhất với endpoint chuyên biệt (nếu service có hàm predict)
                return await self.mobilenet_classifier.predict.remote(image_bytes, selected_task)

            if selected_model in ("efficientnet_b0", "efficientnetb0", "effb0"):
                if self.efficientnet_b0_classifier is None:
                    raise HTTPException(503, "EfficientNetB0 classifier not available")
                # FIX: Sửa thành .predict.remote()
                return await self.efficientnet_b0_classifier.predict.remote(image_bytes, selected_task)

            if selected_model in ("efficientnet_b4", "efficientnetb4", "effb4"):
                if self.efficientnet_b4_classifier is None:
                    raise HTTPException(503, "EfficientNetB4 classifier not available")
                # FIX: Sửa thành .predict.remote()
                return await self.efficientnet_b4_classifier.predict.remote(image_bytes, selected_task)

            raise HTTPException(400, f"Unknown image model '{selected_model}'")

        # JSON branch: text tasks
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "Invalid or missing request body")

        # Prefer explicit deployment selector from query or body
        selected_model = deployment or model or body.get("deployment")
        if not selected_model:
            selected_model = "qwen" if "prompt" in body else ("embedding" if "text" in body else None)
        if not selected_model:
            raise HTTPException(400, "Cannot infer model. Provide 'model' or include 'prompt'/'text'.")

        selected_model = selected_model.lower()

        if selected_model in ("qwen", "llm", "generate", "generation"):
            if self.qwen is None:
                raise HTTPException(503, "Qwen service not available")
            prompt = body.get("prompt")
            if not prompt:
                raise HTTPException(400, "Missing 'prompt'")
            max_tokens = body.get("max_tokens", 256)
            temperature = body.get("temperature", 0.7)
            return await self.qwen.generate.remote({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            })

        if selected_model in ("embedding", "embed", "encoder"):
            if self.embed is None:
                raise HTTPException(503, "Embedding service not available")
            text = body.get("text")
            if not text:
                raise HTTPException(400, "Missing 'text'")
            return await self.embed.encode.remote({"text": text})

        raise HTTPException(400, f"Unknown model '{selected_model}'")


def deploy() -> None:
    """Deploy chỉ Router - các models được deploy từ service files riêng biệt"""
    Router.deploy()
def app(config: dict):
    # Bind toàn bộ deployments để một lệnh có thể deploy tất cả
    from qwen_service import QwenAPI
    from embedding_service import EmbeddingAPI
    from mobilenet_service import MobilenetClassifier
    from efficientnetb0_service import EfficientNetB0Classifier
    from efficientnetb4_service import EfficientNetB4Classifier

    qwen = QwenAPI.bind()
    embedding = EmbeddingAPI.bind()
    mobilenet = MobilenetClassifier.bind()
    eff_b0 = EfficientNetB0Classifier.bind()
    eff_b4 = EfficientNetB4Classifier.bind()

    # Trả về DAG: Router nhận các handles để dùng ở /api/predict
    return Router.bind(
        qwen,
        embedding,
        mobilenet,
        eff_b0,
        eff_b4,
    )