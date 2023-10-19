from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch

from unified_desktop.core.utils.device import get_best_available_device


class UDBase(ABC):
    """Abstract base class for Unified Desktop's (UD) AI models.

    This class serves as the foundational structure for all UD AI modules and should be subclassed.
    Implementing classes must provide definitions for the following abstract methods:
        - _validate_args
        - _load_model
        - _preprocess
        - _predict
        - _postprocess

    Attributes:
        device (Union[str, torch.device]): The computational device to use, either a string or a PyTorch
            device object. If None, defaults to the best available device.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        """Initialize the UDBase object.

        Args:
            device (Union[str, torch.device], optional): Specifies the PyTorch device for running the model.
            Defaults to the best available device.
        """
        self.device = device if device else get_best_available_device()

        self._validate_args()
        self._load_model()

    def _validate_args(self) -> None:
        """Validate initialization arguments."""
        pass

    @abstractmethod
    def _load_model(self) -> None:
        """Load the AI model."""
        raise NotImplementedError

    @abstractmethod
    def _preprocess(self, *args: Any, **kwargs: Any) -> Any:
        """Preprocess input data."""
        raise NotImplementedError

    @abstractmethod
    def _predict(self, *args: Any, **kwargs: Any) -> Any:
        """Generate predictions based on preprocessed data."""
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, *args: Any, **kwargs: Any) -> Any:
        """Postprocess the generated predictions."""
        raise NotImplementedError

    @abstractmethod
    def __call__(*args: Any, **kwargs: Any) -> Any:
        """Perform inference on the input data."""
        raise NotImplementedError
