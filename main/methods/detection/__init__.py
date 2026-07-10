"""仅图像水印检测方法。"""

from main.methods.detection.image_only import (
    ImageOnlyDetectionConfig,
    ImageOnlyDetectionResult,
    detect_image_only_watermark,
)

__all__ = [
    "ImageOnlyDetectionConfig",
    "ImageOnlyDetectionResult",
    "detect_image_only_watermark",
]
