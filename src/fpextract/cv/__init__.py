"""Classical-CV helpers: crop snapping, background normalisation, watermark
removal. All require the ``cv`` extra (``pip install fpextract[cv]``)."""


def require_cv2():
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("OpenCV is required: pip install 'fpextract[cv]'") from e
