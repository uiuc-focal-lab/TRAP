import math
import re

from PIL import Image, ImageDraw, ImageFont



def letter_options(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def choice_options(n: int, *, mode: str) -> list[str]:
    if mode == "letters":
        return letter_options(n)
    if mode == "numbers":
        return [str(i + 1) for i in range(n)]
    raise ValueError(f"Unknown choice mode: {mode!r}")


def extract_choice(text: str, options: list[str]) -> str:
    if not text:
        return "ERROR"
    s = text.strip().upper()

    # If the model echoed the instruction list (e.g., "... A/B/C/D ... ASSISTANT: A"), strip to the tail.
    for marker in ("ASSISTANT:", "<|ASSISTANT|>", "### ASSISTANT:", "ASSISTANT :"):
        if marker in s:
            s = s.split(marker)[-1].strip()

    # Prefer longer options first (e.g., "10" before "1") to avoid partial matches.
    ordered = sorted(options, key=len, reverse=True)
    opt_pattern = "(" + "|".join(map(re.escape, ordered)) + ")"

    # Strict: a single-letter answer (optionally wrapped/punctuated), e.g. "B", "(B)."
    m = re.match(rf"^[\s\(\[\{{]*{opt_pattern}[\s\)\]\}}]*[\s\.\:\,\;\!\?]*$", s)
    if m:
        return m.group(1)

    # Common phrasing, e.g. "Answer: B", "Option C", "Panel D"
    m = re.search(rf"\b(?:ANSWER|OPTION|CHOICE|PANEL|SELECT|PICK|BEST)\b\s*[:\-]?\s*{opt_pattern}\b", s)
    if m:
        return m.group(1)

    # Standalone token match (not embedded in words like 'ANSWER').
    matches = re.findall(rf"(?<![A-Z0-9]){opt_pattern}(?![A-Z0-9])", s)
    uniq = list(dict.fromkeys(matches))
    if len(uniq) == 1:
        return uniq[0]
    return "ERROR"


def _render_label_tile(letter: str, *, box_size: int) -> Image.Image:
    box_size = int(box_size)
    if box_size <= 0:
        raise ValueError("box_size must be positive")

    bg = Image.new("RGB", (box_size, box_size), (0, 0, 0))
    draw = ImageDraw.Draw(bg)

    # Prefer a truetype font if present (much more readable than PIL's tiny default bitmap font).
    font = None
    font_size = max(14, int(box_size * 0.72))
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ):
        try:
            font = ImageFont.truetype(path, size=font_size)
            break
        except Exception:
            font = None

    if font is not None:
        bbox = draw.textbbox((0, 0), letter, font=font, stroke_width=2)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (box_size - w) // 2 - bbox[0]
        y = (box_size - h) // 2 - bbox[1]
        draw.text((x, y), letter, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        return bg

    # Fallback: render with the default bitmap font into a small mask and scale up.
    small = Image.new("L", (48, 48), 0)
    sd = ImageDraw.Draw(small)
    default_font = ImageFont.load_default()
    bbox = sd.textbbox((0, 0), letter, font=default_font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (48 - w) // 2 - bbox[0]
    y = (48 - h) // 2 - bbox[1]
    sd.text((x, y), letter, fill=255, font=default_font)
    mask = small.resize((box_size, box_size), resample=Image.NEAREST)
    bg.paste((255, 255, 255), (0, 0), mask=mask)
    return bg


def concatenate_images_with_labels(images: list[Image.Image], labels: list[str]) -> Image.Image:
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")

    # Paper protocol uses horizontal concatenation with randomized order.
    n = len(images)
    panel_size = min(min(img.width for img in images), min(img.height for img in images))
    panel_size = max(64, min(512, int(panel_size)))
    resized = [img.resize((panel_size, panel_size), Image.LANCZOS) for img in images]

    cols = n
    rows = 1
    out = Image.new("RGB", (cols * panel_size, panel_size), (0, 0, 0))
    draw = ImageDraw.Draw(out)

    pad = max(8, int(panel_size * 0.02))
    box_size = max(64, int(panel_size * 0.14))

    for i, (img, label) in enumerate(zip(resized, labels)):
        r = i // cols
        c = i % cols
        x0 = c * panel_size
        y0 = r * panel_size
        out.paste(img, (x0, y0))

        tile = _render_label_tile(label, box_size=box_size)
        out.paste(tile, (x0 + pad, y0 + pad))

        # Thin border helps the VLM separate panels after resizing.
        draw.rectangle(
            [x0, y0, x0 + panel_size - 1, y0 + panel_size - 1],
            outline=(255, 255, 255),
            width=max(2, panel_size // 256),
        )

    return out


def _demo(out_path: str, n: int) -> None:
    colors = [(200, 60, 60), (60, 200, 60), (60, 60, 200), (220, 180, 60), (180, 60, 220)]
    panels = [Image.new("RGB", (512, 512), colors[i % len(colors)]) for i in range(n)]
    labels = letter_options(n)
    grid = concatenate_images_with_labels(panels, labels)
    grid.save(out_path)
    print(f"Wrote demo grid: {out_path}")

    samples = ["B", "Panel B", "(C).", "I think it's A", "Answer: D", "A/B/C/D", "None"]
    for s in samples:
        print(f"{s!r} -> {extract_choice(s, labels)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check evaluator input rendering and choice parsing.")
    parser.add_argument("--out", type=str, default="./_eval_concat_demo.png")
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()
    _demo(args.out, args.n)
