from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class ModelType(str, Enum):
    YOLO = "yolo"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"

class ValidationResponse(BaseModel):
    valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None