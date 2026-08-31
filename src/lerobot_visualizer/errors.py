"""Domain-specific errors for local dataset access."""


class DatasetAccessError(Exception):
    """Base class for dataset access failures."""


class InvalidDatasetRootError(DatasetAccessError):
    """The configured dataset root is missing or invalid."""


class DatasetMetadataError(DatasetAccessError):
    """Required metadata is missing, malformed, or inconsistent."""


class DatasetReferenceError(DatasetAccessError):
    """A metadata-derived file reference is malformed or unsafe."""


class EpisodeNotFoundError(DatasetAccessError):
    """The requested episode index does not exist."""


class FrameNotFoundError(DatasetAccessError):
    """The requested frame index does not exist in the episode."""


class TimestampLookupError(DatasetAccessError):
    """A timestamp or lookup policy is invalid."""


class TimestampOutOfRangeError(TimestampLookupError):
    """The requested timestamp is outside the episode range."""


class FieldNotFoundError(DatasetAccessError):
    """A requested episode field does not exist."""


class VideoReferenceError(DatasetAccessError):
    """An episode video reference is missing or malformed."""


class VideoNotFoundError(DatasetAccessError):
    """A referenced local video file does not exist."""
