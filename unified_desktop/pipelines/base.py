from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, Sequence, Union

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
        model_id (str): The identifier for the model to be used.
        torch_dtype (torch.dtype): The data type for torch tensors.
        device (Union[str, torch.device]): The computational device to use, either a string or a PyTorch
            device object. If None, defaults to the best available device.
    """

    models_list: ClassVar[Sequence[str]] = ()
    """Class variable containing the list of available models.
    """

    def __init__(
        self,
        model_id: str,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the UDBase object.

        Args:
            model_id (str): The identifier for the model to be used.
            torch_dtype (torch.dtype): The data type for torch tensors.
            device (Union[str, torch.device]): The device to run the model on. If None, the
                default device is used.
        """
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        if torch_dtype is None:
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.device = torch.device(device) if device else get_best_available_device()

        self._validate_args()
        self._load_model_or_pipeline()

    def _validate_args(self) -> None:
        """Validate initialization arguments."""
        if self.model_id not in self.models_list:
            raise ValueError(f"Model {self.model_id} not found; available models: {self.models_list}")

        if self.torch_dtype not in (torch.float16, torch.float32):
            raise ValueError(
                f"Invalid torch_dtype: {self.torch_dtype}. Must be torch.float16 or torch.float32."
            )
        if self.torch_dtype == torch.float16 and not torch.cuda.is_available():
            raise ValueError("torch.float16 is only supported on CUDA devices.")

    @abstractmethod
    def _load_model_or_pipeline(self) -> None:
        """Load the AI model or Transformer pipeline."""
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
