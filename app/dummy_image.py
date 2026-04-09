import numpy as np
from PIL import Image

def create_dummy_image(width=640, height=640) -> np.ndarray:
    """Создаёт RGB тестовое изображение с шумом."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return arr  # RGB