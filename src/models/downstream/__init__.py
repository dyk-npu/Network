"""
Downstream task modules

Contains implementations of various downstream tasks based on multimodal fusion features:
- Classification (classification.py)
- Classification Pipeline (classification_pipeline.py)
"""

from .classification import (
    MultiModalClassifier,
    ClassificationLoss,
    ClassificationMetrics,
    create_classification_model,
    create_classification_loss,
    CAD_PART_CLASSES
)

from .classification_pipeline import (
    MultiModalClassificationPipeline,
    ClassificationTrainer,
    create_classification_pipeline
)

__all__ = [
    # Classification related
    'MultiModalClassifier',
    'ClassificationLoss',
    'ClassificationMetrics',
    'create_classification_model',
    'create_classification_loss',
    'CAD_PART_CLASSES',

    # Classification pipeline
    'MultiModalClassificationPipeline',
    'ClassificationTrainer',
    'create_classification_pipeline'
]