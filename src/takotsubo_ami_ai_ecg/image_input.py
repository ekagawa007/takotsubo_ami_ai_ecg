"""Open only the documented, single-frame raster image formats."""

from .security import require_patched_pillow
from .settings import IMAGE_SIZE, INPUT_POLICIES


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

# Bound the intermediate RGB square before allocating it. A very thin image
# could otherwise expand to an enormous canvas despite a small source file.
MAX_PAD_CANVAS_PIXELS = 40_000_000


def describe_image_geometry(size, input_policy="strict480"):
    """Validate a policy and describe its deterministic geometry without decoding.

    Coordinates are pixels in the source image. Crop boxes use Pillow's
    half-open [left, top, right, bottom] convention. Padding is [left, top,
    right, bottom]; an odd extra pixel is placed on the right or bottom.
    """
    if input_policy not in INPUT_POLICIES:
        raise ImageInputError("Unknown input policy; choose " + ", ".join(INPUT_POLICIES) + ".")
    if (not isinstance(size, (tuple, list)) or len(size) != 2
            or any(type(value) is not int or value < 1 for value in size)):
        raise ImageInputError("Image dimensions must be two positive integers.")
    width, height = size
    if input_policy == "strict480" and (width, height) != (IMAGE_SIZE, IMAGE_SIZE):
        raise ImageInputError(
            "The default requires prepared 480x480 images. Select square-resize, "
            "center-crop or pad explicitly to prepare another image size."
        )
    if input_policy == "square-resize" and width != height:
        raise ImageInputError("square-resize requires a square image; choose center-crop or pad for a rectangle.")
    crop_box = None
    padding = [0, 0, 0, 0]
    before_resize = [width, height]
    operation = "identity"
    if input_policy == "center-crop" and width != height:
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        crop_box = [left, top, left + side, top + side]
        before_resize = [side, side]
        operation = "center_crop_then_resize"
    elif input_policy == "pad" and width != height:
        side = max(width, height)
        if side * side > MAX_PAD_CANVAS_PIXELS:
            raise ImageInputError(
                "Padding would exceed the 40,000,000-pixel intermediate square limit. "
                "Prepare a smaller image or choose another input policy."
            )
        left, top = (side - width) // 2, (side - height) // 2
        padding = [left, top, side - width - left, side - height - top]
        before_resize = [side, side]
        operation = "pad_then_resize"
    resize_applied = before_resize != [IMAGE_SIZE, IMAGE_SIZE]
    if operation == "identity" and resize_applied:
        operation = "resize"
    return {
        "policy": input_policy, "original_size": [width, height],
        "prepared_size": [IMAGE_SIZE, IMAGE_SIZE], "operation": operation,
        "crop_box": crop_box, "padding": padding,
        "padding_color_rgb": [255, 255, 255] if input_policy == "pad" else None,
        "size_before_resize": before_resize, "resize_applied": resize_applied,
        "resize_interpolation": "bilinear" if resize_applied else None,
    }


def prepare_image_geometry(image, input_policy="strict480"):
    """Prepare RGB pixels identically for prediction, preview and Grad-CAM.

    The returned 480x480 image never contains a spatial transform selected
    implicitly. Existing 480x480 images pass through unchanged. This function
    does not overwrite the source, apply EXIF rotation or copy file metadata
    into exported artifacts.
    """
    require_patched_pillow()
    from PIL import Image

    if image.mode != "RGB":
        raise ImageInputError("Image geometry preparation requires RGB pixels.")
    geometry = describe_image_geometry(image.size, input_policy)
    prepared = image
    if geometry["crop_box"] is not None:
        prepared = prepared.crop(tuple(geometry["crop_box"]))
    elif any(geometry["padding"]):
        canvas = Image.new("RGB", tuple(geometry["size_before_resize"]), (255, 255, 255))
        canvas.paste(prepared, tuple(geometry["padding"][:2]))
        prepared = canvas
    if geometry["resize_applied"]:
        prepared = prepared.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
    return prepared


class ImageGeometry:
    """Picklable deterministic geometry transform for DataLoader workers."""

    def __init__(self, input_policy):
        if input_policy not in INPUT_POLICIES:
            raise ImageInputError("Unknown image geometry policy.")
        self.input_policy = input_policy

    def __call__(self, image):
        return prepare_image_geometry(image, self.input_policy)
