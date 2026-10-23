"""Open only the documented, single-frame raster image formats."""

from .security import require_patched_pillow


SUPPORTED_IMAGE_FORMATS = ("PNG", "JPEG", "BMP", "TIFF", "WEBP")


class ImageInputError(ValueError):
    """An expected image rejection that does not include a private pathname."""


def open_supported_image(path, image_number):
    """Return an image for use in a ``with`` block; never try other decoders.

    The decoder restriction applies before file identification, so an EPS
    renamed to .png cannot reach the EPS plugin. Inspecting the decoded format
    also avoids treating a filename extension as evidence of its contents.
    Call this on every actual read, including DataLoader worker reads.
    """
    require_patched_pillow()
    from PIL import Image

    source = None
    try:
        source = Image.open(path, formats=SUPPORTED_IMAGE_FORMATS)
        if source.format not in SUPPORTED_IMAGE_FORMATS:
            raise ImageInputError(f"Input image {image_number} has an unsupported image format.")
        if getattr(source, "n_frames", 1) != 1 or getattr(source, "is_animated", False):
            raise ImageInputError(
                f"Input image {image_number} must contain exactly one frame; "
                "animated and multipage images are not supported."
            )
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        if source is not None:
            source.close()
        if isinstance(error, ImageInputError):
            raise
        raise ImageInputError(
            f"Input image {image_number} is not a readable single-frame "
            "PNG, JPEG, BMP, TIFF or WebP image."
        ) from None
    return source
