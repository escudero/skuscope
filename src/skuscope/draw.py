"""Draws recognized SKUs on the shelf photo."""

from PIL import Image, ImageDraw, ImageFont

from skuscope.pipeline import UNKNOWN, Detection

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#9a6324", "#469990", "#808000",
]
UNKNOWN_COLOR = "#9e9e9e"
MIN_FONT_SIZE = 10


def palette_for(skus: list[str]) -> dict[str, str]:
    """One color per catalog SKU, in catalog order, so colors stay the same across photos."""
    return {sku: PALETTE[i % len(PALETTE)] for i, sku in enumerate(sorted(set(skus)))}


def _fit_label(draw: ImageDraw.ImageDraw, text: str, max_width: float, size: int):
    """Shrinks the font, then truncates the text, until the label fits inside the box."""
    while size > MIN_FONT_SIZE and draw.textlength(text, font=ImageFont.load_default(size=size)) > max_width:
        size -= 1
    font = ImageFont.load_default(size=size)
    while len(text) > 1 and draw.textlength(text, font=font) > max_width:
        text = text[:-2] + "…"
    return text, font


def annotate(image: Image.Image, detections: list[Detection], colors: dict[str, str]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    scale = max(out.size) / 1000
    width = max(2, round(3 * scale))
    base_size = max(MIN_FONT_SIZE, round(16 * scale))

    # Unknowns first, so recognized boxes are drawn on top.
    for d in sorted(detections, key=lambda d: d.sku != UNKNOWN):
        box = (d.box.x1, d.box.y1, d.box.x2, d.box.y2)
        if d.sku == UNKNOWN:
            draw.rectangle(box, outline=UNKNOWN_COLOR, width=max(1, width // 2))
            continue
        color = colors.get(d.sku, UNKNOWN_COLOR)
        draw.rectangle(box, outline=color, width=width)

        # Label inside the top of the box: on a dense shelf there is no free space above it.
        pad = width
        text, font = _fit_label(draw, f"{d.sku} {d.score:.2f}", d.box.x2 - d.box.x1 - 2 * pad, base_size)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        x, y = d.box.x1, d.box.y1
        draw.rectangle((x, y, x + right - left + 2 * pad, y + bottom - top + 2 * pad), fill=color)
        draw.text((x + pad - left, y + pad - top), text, fill="white", font=font)
    return out
