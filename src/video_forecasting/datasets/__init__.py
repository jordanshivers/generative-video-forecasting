"""Dataset loaders for video forecasting experiments."""

from .moving_mnist import MovingMNISTDataset
from .elastic_disks import ElasticDisksDataset

__all__ = ["MovingMNISTDataset", "ElasticDisksDataset"]
