# srag/exceptions.py
"""
SRag typed exception hierarchy.
All public methods raise these instead of raw exceptions.
"""


class SRagError(Exception):
    """Base exception for all SRag errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self):
        return f"{self.__class__.__name__}({self.message!r})"


# ── Retrieval errors ──────────────────────────────────────────────────────────

class SRagFetchError(SRagError):
    """Raised when a URL or source cannot be fetched."""
    pass


class SRagTimeoutError(SRagFetchError):
    """Raised when a fetch times out."""
    pass


class SRagBlockedError(SRagFetchError):
    """Raised when a domain blocks the request (403, 429)."""
    pass


# ── Quality errors ────────────────────────────────────────────────────────────

class SRagQualityError(SRagError):
    """Raised when content fails the quality gate."""
    def __init__(self, message: str, pass_rate: float = 0.0, passed_chunks: int = 0):
        super().__init__(message, {
            "pass_rate":     pass_rate,
            "passed_chunks": passed_chunks,
        })
        self.pass_rate     = pass_rate
        self.passed_chunks = passed_chunks


class SRagNoContentError(SRagError):
    """Raised when no usable content is found after scraping."""
    pass


# ── Index errors ──────────────────────────────────────────────────────────────

class SRagIndexError(SRagError):
    """Raised when indexing into LanceDB fails."""
    pass


class SRagSessionNotFoundError(SRagError):
    """Raised when a session is queried but doesn't exist."""
    def __init__(self, session: str):
        super().__init__(
            f"Session '{session}' not found. Run `srag index` first.",
            {"session": session},
        )
        self.session = session


# ── Ingest errors ─────────────────────────────────────────────────────────────

class SRagIngestError(SRagError):
    """Raised when local file or database ingestion fails."""
    def __init__(self, message: str, source: str = ""):
        super().__init__(message, {"source": source})
        self.source = source


class SRagUnsupportedFormatError(SRagIngestError):
    """Raised when an unsupported file format is passed to the ingestor."""
    def __init__(self, ext: str):
        super().__init__(
            f"Unsupported file format: '{ext}'. "
            f"Supported: .pdf, .docx, .txt, .csv, .json",
            source=ext,
        )


class SRagMissingDependencyError(SRagIngestError):
    """Raised when an optional dependency is missing."""
    def __init__(self, package: str, install_extra: str):
        super().__init__(
            f"'{package}' is required for this operation. "
            f"Install it with: pip install srag[{install_extra}]",
        )


# ── Config errors ─────────────────────────────────────────────────────────────

class SRagConfigError(SRagError):
    """Raised when SRagConfig has invalid values."""
    pass