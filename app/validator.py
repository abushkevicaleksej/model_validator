import io
import tempfile
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image

# Фреймворки
import torch
from ultralytics import YOLO
import onnxruntime as ort
import tensorflow as tf

from .dummy_image import create_dummy_image

class ModelValidator:

    @staticmethod
    def _to_numpy(image_bytes: bytes) -> np.ndarray:
        """Конвертирует байты изображения в numpy array (RGB)."""
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return np.array(img)

    @staticmethod
    def _check_output_format(predictions: Any) -> bool:
        try:
            if hasattr(predictions, 'boxes') and hasattr(predictions.boxes, 'xyxy'):
                boxes = predictions.boxes.xyxy.cpu().numpy()
                confs = predictions.boxes.conf.cpu().numpy()
                cls_ids = predictions.boxes.cls.cpu().numpy()
                return len(boxes) >= 0
            if isinstance(predictions, (list, tuple)) and len(predictions) > 0:
                first = predictions[0]
                if isinstance(first, dict):
                    keys = set(first.keys())
                    if keys == {'boxes', 'labels', 'scores'}:
                        if all(k in first for k in ('boxes', 'labels', 'scores')):
                            return True
                    if 'boxes' in keys and 'scores' in keys:
                        return True
                return False
            elif isinstance(predictions, (list, tuple)):
                for pred in predictions:
                    if isinstance(pred, dict):
                        if not ('bbox' in pred or 'box' in pred or 'x1' in pred):
                            return False
                        if not ('score' in pred or 'confidence' in pred):
                            return False
                        if not ('label' in pred or 'class_id' in pred):
                            return False
                    elif isinstance(pred, (list, tuple)) and len(pred) >= 6:
                        continue
                    else:
                        return False
                return True
            elif isinstance(predictions, np.ndarray) and predictions.ndim == 2:
                # Массив Nx6 (или Nx5)
                return predictions.shape[1] >= 5
            else:
                return False
        except Exception:
            return False

    @classmethod
    def validate_yolov8(cls, model_bytes: bytes, test_image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        try:
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                tmp.write(model_bytes)
                tmp_path = tmp.name

            model = YOLO(tmp_path)
            # Тестовое изображение
            if test_image_bytes:
                img = cls._to_numpy(test_image_bytes)
            else:
                img = create_dummy_image()

            results = model(img, verbose=False)
            if not results:
                return {"valid": False, "message": "Модель не вернула результатов"}
            pred = results[0]
            is_valid = cls._check_output_format(pred)
            details = {
                "num_detections": len(pred.boxes) if pred.boxes else 0,
                "output_format": "YOLO Results object"
            }
            return {"valid": is_valid, "message": "OK" if is_valid else "Неверный формат выхода", "details": details}
        except Exception as e:
            return {"valid": False, "message": f"Ошибка валидации YOLO: {str(e)}"}

    @classmethod
    def validate_pytorch(cls, model_bytes: bytes, test_image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        try:
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
                tmp.write(model_bytes)
                tmp_path = tmp.name

            model = torch.load(tmp_path, map_location='cpu')
            if isinstance(model, dict):
                return {"valid": False, "message": "Передан state_dict без архитектуры. Укажите полную модель."}

            model.eval()

            if test_image_bytes:
                img = cls._to_numpy(test_image_bytes)
                img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0
            else:
                img_tensor = torch.randn(3, 640, 640)

            with torch.no_grad():
                out = model([img_tensor])

            is_valid = cls._check_output_format(out)
            return {"valid": is_valid, "message": "OK" if is_valid else "Неверный формат выхода", "details": {"output_shape": str(out.shape) if hasattr(out, 'shape') else None}}
        except Exception as e:
            return {"valid": False, "message": f"Ошибка валидации PyTorch: {str(e)}"}

    @classmethod
    def validate_onnx(cls, model_bytes: bytes, test_image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        try:
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
                tmp.write(model_bytes)
                tmp_path = tmp.name

            sess = ort.InferenceSession(tmp_path)
            input_name = sess.get_inputs()[0].name
            if test_image_bytes:
                img = cls._to_numpy(test_image_bytes)
                input_tensor = img.transpose(2,0,1)[None, ...].astype(np.float32) / 255.0
            else:
                input_tensor = np.random.randn(1, 3, 640, 640).astype(np.float32)

            outputs = sess.run(None, {input_name: input_tensor})
            is_valid = cls._check_output_format(outputs)
            return {"valid": is_valid, "message": "OK" if is_valid else "Неверный формат выхода", "details": {"num_outputs": len(outputs)}}
        except Exception as e:
            return {"valid": False, "message": f"Ошибка валидации ONNX: {str(e)}"}

    @classmethod
    def validate_tensorflow(cls, model_bytes: bytes, test_image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = f"{tmpdir}/model.h5"
                with open(model_path, 'wb') as f:
                    f.write(model_bytes)
                model = tf.keras.models.load_model(model_path)

                if test_image_bytes:
                    img = cls._to_numpy(test_image_bytes)
                    input_tensor = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
                else:
                    input_tensor = np.random.randn(1, 640, 640, 3).astype(np.float32)

                predictions = model.predict(input_tensor, verbose=0)
                is_valid = cls._check_output_format(predictions)
                return {"valid": is_valid, "message": "OK" if is_valid else "Неверный формат выхода", "details": {"output_shape": str(predictions.shape)}}
        except Exception as e:
            return {"valid": False, "message": f"Ошибка валидации TensorFlow: {str(e)}"}