"""Package-specific errors with actionable messages."""


class FarmerRLError(RuntimeError):
    """Base error for this package."""


class OptionalDependencyError(FarmerRLError):
    """Raised when an explicitly requested optional integration is unavailable."""


class SeatSafetyError(FarmerRLError, ValueError):
    """Raised before data from one acting seat can be labelled as another seat."""


class InvalidActionError(FarmerRLError, ValueError):
    """Raised when an action does not satisfy the public action contract."""
