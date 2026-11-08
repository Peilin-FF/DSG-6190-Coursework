"""Image loading and encoding for OpenAI-compatible multimodal requests."""

import base64
import functools
import io
import os

from PIL import Image, ImageOps


def load_rgb(path: str) -> Image.Image:
    """Open an image, apply its EXIF orientation and flatten transparency onto white."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def resize_max_side(im: Image.Image, max_side: int) -> Image.Image:
    w, h = im.size
    side = max(w, h)
    if not max_side or side <= max_side:
        return im
    r = max_side / side
    return im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.BICUBIC)


@functools.lru_cache(maxsize=8192)
def encode_image(path: str, max_side: int = 1024) -> str:
    """Return a data URL. Photos are sent as JPEG (q=95), diagrams and renders as PNG."""
    im = resize_max_side(load_rgb(path), max_side)
    buf = io.BytesIO()
    if path.lower().endswith((".jpg", ".jpeg")):
        im.save(buf, "JPEG", quality=95)
        mime = "image/jpeg"
    else:
        im.save(buf, "PNG")
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def image_fingerprint(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.abspath(path)}:{st.st_size}"
