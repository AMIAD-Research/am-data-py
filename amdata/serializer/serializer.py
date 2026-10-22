from typing import Dict, List, Any
from abc import ABC, abstractmethod

import ast

class Serializer(ABC):
    """
    Abstract class for serializers.

    Serializers are used to write and load complex objects.
    """

    def __init__(self,
        param_template: Dict[str,str]=None,
        **kwargs
    ) -> None:
        """
        Args:
            param_template (Dict, optional): template used to extract arguments of the __init__ method from information of a corpus. Defaults to None.
        """

        self._param_template = param_template

    def get_param_template(self
    ) -> None:
        """
        Getter for param_template.
        """
        return self._param_template

    def set_param_template(self,
        param_template: Dict[str,str]
    ) -> None:
        """
        Setter for param_template.
        """
        self._param_template = param_template

    def __call__(self,
        **kwargs
    ) -> object:

        return self.deserialize(**kwargs)

    @abstractmethod
    def serialize(self, obj: object) -> str:
        """Convert the object to a serialized string format."""
        pass

    @abstractmethod
    def deserialize(self, data: str) -> object:
        """Reconstruct the object from the serialized string."""
        pass

class StringSerializer(Serializer):
    """
    Serializes objects to their string representation (via ``repr``)
    and deserializes strings back to Python objects (via ``ast.literal_eval``).

    Only safely deserializes strings containing lists of strings;
    raises ``ValueError`` otherwise.

    Inherits from ``Serializer`` and delegates ``param_template`` handling.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the string serializer.

        Args:
            **kwargs: arguments forwarded to the parent ``Serializer.__init__``.
        """
        super().__init__(**kwargs)

    def serialize(self, obj: Any) -> str:
        """
        Convert an object to its string representation.

        Uses ``repr()`` to preserve object structure.

        Args:
            obj: the object to serialize.

        Returns:
            str: the string representation of ``obj``.
        """
        return repr(obj)  # or str(obj), depending on preference

    def deserialize(self, input: str) -> List[str]:
        """
        Parse a string back into a Python list of strings.

        Uses ``ast.literal_eval`` for safe evaluation. Raises
        ``ValueError`` if the result is not a list of strings.

        Args:
            input: a string containing a Python list literal,
                e.g. ``"['hello', 'world']"``.

        Returns:
            List[str]: the parsed list of strings.

        Raises:
            ValueError: if the parsed result is not a list of strings.
        """
        output = ast.literal_eval(input)
        if not isinstance(output, list) or not all(isinstance(i, str) for i in output):
            raise ValueError("Deserialized input is not a List[str]")
        return output