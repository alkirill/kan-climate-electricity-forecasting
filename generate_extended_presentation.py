from pathlib import Path

import fitz
from PIL import Image


ROOT = Path("/Users/kirill/Desktop/ВКР ИТМО")
OLD_PDF = ROOT / "НИР_2025_Алехин_Кирилл (3).pdf"
ADDON_PDF = ROOT / "НИР_2025_дополнительные_слайды.pdf"
OUT_DIR = ROOT / "presentation_extended_assets"
OUT_DIR.mkdir(exist_ok=True)
OUT_PDF = ROOT / "НИР_2025_расширенная_презентация.pdf"


def pdf_pages_to_images(pdf_path: Path, start: int = 0, end: int | None = None, scale: float = 1.8):
    doc = fitz.open(str(pdf_path))
    page_count = doc.pageCount
    pages = range(start, page_count if end is None else end)
    images = []

    for i in pages:
        page = doc.loadPage(i)
        pix = page.getPixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img_path = OUT_DIR / f"{pdf_path.stem}_p{i + 1}.png"
        pix.writePNG(str(img_path))
        images.append(Image.open(img_path).convert("RGB"))

    return images


def main():
    slides = []

    # Keep the original presentation body intact.
    slides.extend(pdf_pages_to_images(OLD_PDF, start=0, end=18))

    # Insert the new expanded electricity block.
    slides.extend(pdf_pages_to_images(ADDON_PDF))

    # Move the original thank-you page to the very end.
    slides.extend(pdf_pages_to_images(OLD_PDF, start=18, end=19))

    slides[0].save(OUT_PDF, save_all=True, append_images=slides[1:])
    print(OUT_PDF)
    print(f"total_slides={len(slides)}")


if __name__ == "__main__":
    main()
