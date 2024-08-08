from enum import Enum, auto


class ModelType(Enum):
    ENCODER_ONLY = auto()
    ENCODER_DECODER = auto()

    @classmethod
    def from_string(cls, model_type_str: str):
        """
        Convert a string representation to a ModelType enum.

        Args:
            model_type_str (str): String representation of the model type.

        Returns:
            ModelType: Corresponding ModelType enum value.

        Raises:
            ValueError: If the input string doesn't match any ModelType.
        """
        model_type_str = model_type_str.upper().replace("-", "_")
        try:
            return cls[model_type_str]
        except KeyError:
            raise ValueError(
                f"Invalid model type: {model_type_str}. "
                f"Valid options are: {', '.join(cls.__members__.keys())}"
            )

    def __str__(self):
        return self.name.lower().replace("_", "-")


# Usage examples
print(ModelType.ENCODER_ONLY)  # Output: ModelType.ENCODER_ONLY
print(str(ModelType.ENCODER_DECODER))  # Output: encoder-decoder
print(ModelType.from_string("encoder-only"))  # Output: ModelType.ENCODER_ONLY
