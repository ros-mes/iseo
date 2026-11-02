"""Exceptions raised by the components of ISEO."""


class ModelResolutionFailedError(Exception):
    """Raised when a programming model solved by a component of ISEO does not provide a usable solution.

    Args:
        message: Description of the failure.
    """

    def __init__(self, message: str):
        super().__init__(message)
