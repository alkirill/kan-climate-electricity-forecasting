from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import fitz


ROOT = Path("/Users/kirill/Desktop/ВКР ИТМО")
OUT_DIR = ROOT / "presentation_final_assets"
OUT_DIR.mkdir(exist_ok=True)

W, H = 1920, 1080
TITLE_Y = 55
LEFT_X = 70
BODY_X = 105
BODY_Y = 190

FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"

F_TITLE = ImageFont.truetype(FONT_BOLD, 54)
F_TEXT = ImageFont.truetype(FONT_REG, 30)
F_TEXT_BOLD = ImageFont.truetype(FONT_BOLD, 30)
F_SMALL = ImageFont.truetype(FONT_REG, 24)
F_LOGO = ImageFont.truetype(FONT_BOLD, 54)
F_PAGE = ImageFont.truetype(FONT_REG, 26)


def render_pdf_page(pdf_path: Path, page_idx: int, out_path: Path, scale: float = 1.6):
    doc = fitz.open(str(pdf_path))
    page = doc.loadPage(page_idx)
    pix = page.getPixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    pix.writePNG(str(out_path))
    return out_path


def draw_wrapped(draw, text, xy, font, max_width, fill="black", line_gap=8):
    x, y = xy
    words = text.split()
    line = ""
    lines = []
    for w in words:
        cand = (line + " " + w).strip()
        if draw.textbbox((0, 0), cand, font=font)[2] <= max_width:
            line = cand
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    for ln in lines:
        draw.text((x, y), ln, fill=fill, font=font)
        y += font.size + line_gap
    return y


def draw_bullets(draw, bullets, x, y, max_width):
    for bullet in bullets:
        draw.text((x - 35, y + 2), "•", fill="black", font=F_TEXT_BOLD)
        y = draw_wrapped(draw, bullet, (x, y), F_TEXT, max_width)
        y += 18
    return y


def make_white_slide(page_no: int, title: str, bullets=None, image=None, image_box=None, footer_note=None):
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((LEFT_X, TITLE_Y), title, fill="black", font=F_TITLE)
    draw.text((W - 310, 55), "itmo", fill="black", font=F_LOGO)
    draw.text((W - 80, H - 55), str(page_no), fill="black", font=F_PAGE)

    if bullets:
        max_width = 1500 if image is None else 820
        draw_bullets(draw, bullets, BODY_X, BODY_Y, max_width)

    if image is not None:
        box = image_box or (1010, 210, 1820, 920)
        draw.rounded_rectangle(box, radius=8, outline=(180, 180, 180), width=2)
        plot = image.copy()
        plot.thumbnail((box[2] - box[0] - 40, box[3] - box[1] - 40))
        px = box[0] + ((box[2] - box[0]) - plot.width) // 2
        py = box[1] + ((box[3] - box[1]) - plot.height) // 2
        img.paste(plot, (px, py))

    if footer_note:
        draw_wrapped(draw, footer_note, (BODY_X, H - 145), F_SMALL, 1500, fill=(80, 80, 80), line_gap=6)

    return img


def main():
    old_pdf = ROOT / "НИР_2025_Алехин_Кирилл (3).pdf"
    title_png = render_pdf_page(old_pdf, 0, OUT_DIR / "title_page.png")
    thanks_png = render_pdf_page(old_pdf, 18, OUT_DIR / "thanks_page.png")

    imgs = {
        "title": Image.open(title_png).convert("RGB"),
        "thanks": Image.open(thanks_png).convert("RGB"),
        "climate_rmse": Image.open(ROOT / "presentation_addon_assets" / "generated_slides" / "slide_13.png").convert("RGB") if (ROOT / "presentation_addon_assets" / "generated_slides" / "slide_13.png").exists() else None,
        "price_by_hour": Image.open(ROOT / "analysis_outputs" / "price_by_hour.png").convert("RGB"),
        "actual_pred": Image.open(ROOT / "analysis_outputs" / "russian_electricity_real_kan_actual_vs_pred.png").convert("RGB"),
        "rmse_strong": Image.open(ROOT / "analysis_outputs" / "russian_electricity_stronger_kan_rmse.png").convert("RGB"),
        "train_strong": Image.open(ROOT / "analysis_outputs" / "russian_electricity_stronger_kan_training.png").convert("RGB"),
        "kan_phi": Image.open(ROOT / "analysis_outputs" / "kan_phi_functions.png").convert("RGB"),
        "hybrid_phi": Image.open(ROOT / "analysis_outputs" / "hybridkan_phi_functions.png").convert("RGB"),
        "region_rmse": Image.open(ROOT / "analysis_outputs" / "russian_electricity_real_kan_top_region_rmse.png").convert("RGB"),
    }

    slides = []
    slides.append(imgs["title"])
    slides.append(
        make_white_slide(
            2,
            "Проблема и цель исследования",
            [
                "Современные задачи прогнозирования требуют одновременно высокой точности и интерпретируемости.",
                "KAN интересен как нелинейная архитектура, в которой можно анализировать вклад каналов через phi-функции.",
                "Цель работы: проверить применимость KAN и Hybrid KAN на задачах прогнозирования разной природы.",
            ],
            footer_note="Исследование выполнено на двух кейсах: климатические данные и данные по электроэнергии РФ.",
        )
    )
    slides.append(
        make_white_slide(
            3,
            "Дизайн исследования",
            [
                "Кейс 1: климатический многомерный временной ряд UCI Air Quality.",
                "Кейс 2: панельный leakage-safe датасет по электроэнергии РФ для 34 регионов.",
                "Сравниваются KAN, Hybrid KAN и baseline-модели.",
                "Отдельно анализируется интерпретируемость через phi-функции и вклад признаков.",
            ],
        )
    )
    slides.append(
        make_white_slide(
            4,
            "Кейс 1: климатические данные",
            [
                "Использован UCI Air Quality Dataset: CO(GT), NO2(GT), C6H6(GT), T, RH.",
                "Постановка: multivariate forecasting со скользящими окнами.",
                "Сравнение: KAN, Hybrid KAN, LSTM, MLP.",
                "Также проверялись устойчивость к шуму, пропускам и уменьшению обучающей выборки.",
            ],
        )
    )
    slides.append(
        make_white_slide(
            5,
            "Кейс 1: результаты и вывод",
            [
                "На климатических данных лучшей моделью оказался Hybrid KAN.",
                "Hybrid KAN показал лучший RMSE среди сравниваемых нейросетевых моделей.",
                "Чистый KAN оказался менее точным, но дал более прозрачную интерпретацию.",
            ],
            image=imgs["hybrid_phi"],
            footer_note="Основной вывод по климату: Hybrid KAN хорошо работает на многомерных временных рядах и сохраняет полезную интерпретируемость.",
        )
    )
    slides.append(
        make_white_slide(
            6,
            "Кейс 2: данные по электроэнергии РФ",
            [
                "Датасет собран из трёх источников: почасовые торги и перетоки, суточные потери, месячные тарифные показатели.",
                "Построена leakage-safe панель: 99 278 наблюдений, 34 региона.",
                "Целевая переменная: прогноз цены электроэнергии на один час вперёд.",
                "Ключевые свойства данных: суточная сезонность, высокая инерционность, региональная неоднородность.",
            ],
            image=imgs["price_by_hour"],
        )
    )
    slides.append(
        make_white_slide(
            7,
            "Кейс 2: базовая leakage-safe постановка",
            [
                "В базовой постановке сравнивались линейные модели, HistGradientBoosting, KAN и Hybrid KAN.",
                "Лучший overall baseline: HistGradientBoosting, RMSE = 101.32.",
                "Чистый KAN дал RMSE = 149.51, исходный Hybrid KAN — RMSE = 181.23.",
                "Это показало, что задачу нужно перестроить под KAN, а не сравнивать в сыром виде.",
            ],
            image=imgs["actual_pred"],
        )
    )
    slides.append(
        make_white_slide(
            8,
            "Как была усилена KAN-модель",
            [
                "Вместо абсолютной цены модель предсказывает delta price = price(t+1) - price(t).",
                "Регион кодируется через trainable embedding, а не только через one-hot.",
                "Набор признаков сокращён до компактного и наиболее информативного пространства.",
                "Добавлена residual-схема: линейная модель объясняет базовую часть, KAN учит остаток.",
            ],
            image=imgs["train_strong"],
        )
    )
    slides.append(
        make_white_slide(
            9,
            "Кейс 2: результаты после усиления",
            [
                "Лучший baseline overall: HGBDelta, RMSE = 102.56.",
                "Лучшая KAN-модель: HybridKANEmbedDelta, RMSE = 107.57.",
                "ResidualRidgeKAN также показал сильный результат: RMSE = 108.56.",
                "По сравнению с исходным Hybrid KAN качество улучшилось с 181.23 до 107.57 по RMSE.",
            ],
            image=imgs["rmse_strong"],
        )
    )
    slides.append(
        make_white_slide(
            10,
            "Что удалось интерпретировать",
            [
                "В чистом KAN можно анализировать phi-функции и вклад отдельных каналов признаков.",
                "В гибридных моделях интерпретация сохраняется частично: итоговый прогноз складывается из линейной и KAN-компоненты.",
                "На электроэнергии наиболее значимыми оказались ценовая история, календарные признаки и история нагрузки.",
            ],
            image=imgs["kan_phi"],
            footer_note="Следовательно, KAN в работе выступает не только как модель прогноза, но и как инструмент анализа нелинейных зависимостей.",
        )
    )
    slides.append(
        make_white_slide(
            11,
            "Суммарное сравнение двух кейсов",
            [
                "На климатических данных лучшей моделью оказался Hybrid KAN.",
                "На данных по электроэнергии лучшим overall baseline стал HGBDelta.",
                "После усиления HybridKANEmbedDelta почти догнал лучший baseline и стал лучшей KAN-моделью энергетического кейса.",
                "Это показывает, что сильные стороны KAN зависят от природы данных и постановки задачи.",
            ],
            image=imgs["region_rmse"],
        )
    )
    slides.append(
        make_white_slide(
            12,
            "Итоговые выводы",
            [
                "KAN и Hybrid KAN применимы к задачам прогнозирования и дают интерпретируемые нелинейные зависимости.",
                "Гибридная архитектура особенно эффективна на многомерных временных рядах климатического типа.",
                "На сложных панельных энергетических данных tree-based baseline остаются сильнее, но разрыв удалось резко сократить.",
                "Главный вклад работы: показаны и сильные стороны KAN, и границы его применимости на реальных данных.",
            ],
        )
    )
    slides.append(imgs["thanks"])

    paths = []
    for i, slide in enumerate(slides, start=1):
        path = OUT_DIR / f"final_slide_{i}.png"
        slide.save(path)
        paths.append(path)

    pdf_path = ROOT / "НИР_2025_финальная_презентация.pdf"
    slide_imgs = [Image.open(p).convert("RGB") for p in paths]
    slide_imgs[0].save(pdf_path, save_all=True, append_images=slide_imgs[1:])
    print(pdf_path)


if __name__ == "__main__":
    main()
