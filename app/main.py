from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
import numpy as np

from .models import ModelType, ValidationResponse
from .validator import ModelValidator

app = FastAPI(title="Model Validator for Object Detection")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/validate", response_model=ValidationResponse)
async def validate_model(
    model_file: UploadFile = File(..., description="Файл модели (.pt, .pth, .onnx, .h5, .zip)"),
    model_type: ModelType = Form(..., description="Тип модели"),
    test_image: Optional[UploadFile] = File(None, description="Опциональное тестовое изображение"),
):
    model_bytes = await model_file.read()
    if not model_bytes:
        raise HTTPException(status_code=400, detail="Пустой файл модели")

    test_image_bytes = None
    if test_image:
        test_image_bytes = await test_image.read()

    if model_type == ModelType.YOLO:
        result = ModelValidator.validate_yolov8(model_bytes, test_image_bytes)
    elif model_type == ModelType.PYTORCH:
        result = ModelValidator.validate_pytorch(model_bytes, test_image_bytes)
    elif model_type == ModelType.ONNX:
        result = ModelValidator.validate_onnx(model_bytes, test_image_bytes)
    elif model_type == ModelType.TENSORFLOW:
        result = ModelValidator.validate_tensorflow(model_bytes, test_image_bytes)
    else:
        raise HTTPException(status_code=400, detail=f"Неподдерживаемый тип модели: {model_type}")

    return ValidationResponse(**result)