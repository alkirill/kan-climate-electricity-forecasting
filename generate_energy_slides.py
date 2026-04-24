from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


W, H = 1920, 1080
MARGIN_X = 70
TITLE_Y = 55
BODY_X = 105
BODY_Y = 190
BULLET_GAP = 20

ROOT = Path("/Users/kirill/Desktop/ВКР ИТМО")
OUT = ROOT / "presentation_addon_assets" / "generated_slides"
OUT.mkdir(parents=True, exist_ok=True)

font_bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
font_reg = "/System/Library/Fonts/Supplemental/Arial.ttf"
TITLE = ImageFont.truetype(font_bold, 54)
TEXT = ImageFont.truetype(font_reg, 29)
TEXT_BOLD = ImageFont.truetype(font_bold, 29)
LOGO = ImageFont.truetype(font_bold, 54)
PAGE = ImageFont.truetype(font_reg, 26)


def draw_wrapped(draw, text, xy, font, max_width, fill="black"):
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
        y += font.size + 8
    return y


def draw_bullets(draw, bullets, start_y, max_width):
    y = start_y
    for bullet in bullets:
        draw.text((BODY_X - 35, y + 2), "•", fill="black", font=TEXT_BOLD)
        y = draw_wrapped(draw, bullet, (BODY_X, y), TEXT, max_width)
        y += BULLET_GAP
    return y


def make_slide(page, title, bullets=None, image=None, image_box=None):
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((MARGIN_X, TITLE_Y), title, fill="black", font=TITLE)
    draw.text((W - 310, 55), "itmo", fill="black", font=LOGO)
    draw.text((W - 80, H - 55), str(page), fill="black", font=PAGE)

    if bullets:
        max_width = 1500 if image is None else 860
        draw_bullets(draw, bullets, BODY_Y, max_width)

    if image is not None:
        box = image_box or (1010, 210, 1820, 920)
        draw.rounded_rectangle(box, radius=8, outline=(180, 180, 180), width=2)
        plot = image.copy()
        plot.thumbnail((box[2] - box[0] - 40, box[3] - box[1] - 40))
        px = box[0] + ((box[2] - box[0]) - plot.width) // 2
        py = box[1] + ((box[3] - box[1]) - plot.height) // 2
        img.paste(plot, (px, py))

    path = OUT / f"slide_{page}.png"
    img.save(path)
    return path


def main():
    imgs = {
        "price_by_hour": Image.open(ROOT / "analysis_outputs" / "price_by_hour.png").convert("RGB"),
        "top_regions": Image.open(ROOT / "analysis_outputs" / "top_regions_mean_price.png").convert("RGB"),
        "actual_pred": Image.open(ROOT / "analysis_outputs" / "russian_electricity_real_kan_actual_vs_pred.png").convert("RGB"),
        "real_rmse": Image.open(ROOT / "analysis_outputs" / "russian_electricity_real_kan_top_region_rmse.png").convert("RGB"),
        "strong_rmse": Image.open(ROOT / "analysis_outputs" / "russian_electricity_stronger_kan_rmse.png").convert("RGB"),
        "strong_train": Image.open(ROOT / "analysis_outputs" / "russian_electricity_stronger_kan_training.png").convert("RGB"),
        "kan_phi": Image.open(ROOT / "analysis_outputs" / "kan_phi_functions.png").convert("RGB"),
        "hybrid_phi": Image.open(ROOT / "analysis_outputs" / "hybridkan_phi_functions.png").convert("RGB"),
    }

    slide_paths = []
    slide_paths.append(
        make_slide(
            19,
            "Что сделано за год",
            [
                "Исследованы архитектуры KAN и Hybrid KAN в задачах прогнозирования.",
                "Проведён цикл экспериментов на климатическом многомерном временном ряде UCI Air Quality.",
                "Построен и проанализирован панельный leakage-safe датасет по электроэнергии РФ для 34 регионов.",
                "Разработана усиленная KAN-постановка: delta target, region embedding и residual learning.",
            ],
        )
    )
    slide_paths.append(
        make_slide(
            20,
            "Анализ данных: электроэнергия РФ",
            [
                "Датасет собран из трёх источников: почасовые торги и перетоки, суточные потери, месячные тарифные показатели.",
                "Итоговая leakage-safe панель: 99 278 наблюдений, 34 региона, период с 30.08.2023 по 31.12.2023.",
                "Целевая переменная: прогноз средневзвешенной цены покупки электроэнергии на один час вперёд.",
                "Задача существенно сложнее климатического кейса из-за межрегиональной неоднородности и таблично-панельной структуры.",
            ],
            imgs["top_regions"],
        )
    )
    slide_paths.append(
        make_slide(
            21,
            "EDA по электроэнергии",
            [
                "В данных наблюдается выраженная суточная сезонность цены.",
                "Максимальные средние цены приходятся на дневные и вечерние часы нагрузки.",
                "Цена обладает высокой инерционностью: лаговые признаки оказываются ключевыми для прогноза.",
                "Между регионами наблюдаются заметные различия по среднему уровню цены и вариативности.",
            ],
            imgs["price_by_hour"],
        )
    )
    slide_paths.append(
        make_slide(
            22,
            "Постановка задачи и модели",
            [
                "Базовая leakage-safe постановка: прогноз цены на t+1 без утечки информации из будущего.",
                "Суточные потери используются только с лагом 1 день, месячные тарифные признаки — только с лагом 1 месяц.",
                "Сравнивались baseline-модели, линейные модели, HistGradientBoosting, чистый KAN и Hybrid KAN.",
                "Для KAN использовались настоящие PyTorch-реализации с визуализацией phi-функций.",
            ],
            imgs["actual_pred"],
        )
    )
    slide_paths.append(
        make_slide(
            23,
            "Результаты базового эксперимента",
            [
                "Лучшей моделью overall в исходной leakage-safe постановке оказался HistGradientBoosting: RMSE = 101.32.",
                "Чистый KAN показал RMSE = 149.51 и превзошёл только наивные сезонные baseline.",
                "Исходный HybridKAN оказался слабым: RMSE = 181.23.",
                "Это показало, что для электроэнергии нужна более удачная постановка задачи под KAN.",
            ],
            imgs["real_rmse"],
        )
    )
    slide_paths.append(
        make_slide(
            24,
            "Усиление KAN-модели",
            [
                "Целевая переменная изменена: модель предсказывает delta price = price(t+1) - price(t).",
                "Категориальный признак региона кодируется через trainable region embedding.",
                "Пространство признаков сокращено до более компактного и информативного набора.",
                "Добавлена residual-схема: Ridge прогнозирует базовую часть, KAN доучивает остаток.",
            ],
            imgs["strong_train"],
        )
    )
    slide_paths.append(
        make_slide(
            25,
            "Результаты после усиления",
            [
                "Лучший baseline overall: HGBDelta, RMSE = 102.56.",
                "Лучшая модель семейства KAN: HybridKANEmbedDelta, RMSE = 107.57.",
                "ResidualRidgeKAN также показал сильный результат: RMSE = 108.56.",
                "По сравнению с исходным HybridKAN качество улучшилось с RMSE 181.23 до 107.57.",
            ],
            imgs["strong_rmse"],
        )
    )
    slide_paths.append(
        make_slide(
            26,
            "Интерпретация KAN на электроэнергии",
            [
                "Для KAN и Hybrid KAN визуализированы phi-функции первого слоя.",
                "Наиболее значимыми оказываются признаки ценовой истории, календаря и нагрузки.",
                "Чистый KAN интерпретируется через каналы и phi-функции напрямую.",
                "В Hybrid KAN интерпретация сохраняется частично, так как итоговый прогноз складывается из линейной и KAN-компоненты.",
            ],
            imgs["kan_phi"],
        )
    )
    slide_paths.append(
        make_slide(
            27,
            "Интерпретация усиленной гибридной модели",
            [
                "После усиления модель стала точнее, но интерпретация не исчезла полностью.",
                "Нелинейная часть HybridKANEmbedDelta по-прежнему позволяет анализировать форму phi-функций.",
                "Сильнее всего на прогноз влияют канал ценовой истории, календарный канал и история нагрузки.",
                "Таким образом, KAN выступает как интерпретируемая нелинейная альтернатива black-box моделям.",
            ],
            imgs["hybrid_phi"],
        )
    )
    slide_paths.append(
        make_slide(
            28,
            "Итоговые выводы по работе",
            [
                "На климатических данных лучшей моделью оказалась Hybrid KAN.",
                "На электроэнергии лучшим overall baseline стала модель HGBDelta.",
                "После усиления HybridKANEmbedDelta почти догнал лучший baseline и стал лучшей KAN-моделью для энергетического кейса.",
                "Главный вклад работы: показаны как сильные стороны KAN, так и границы его применимости на реальных данных.",
            ],
        )
    )

    slide_imgs = [Image.open(p).convert("RGB") for p in slide_paths]
    pdf_path = ROOT / "НИР_2025_дополнительные_слайды.pdf"
    slide_imgs[0].save(pdf_path, save_all=True, append_images=slide_imgs[1:])
    print(pdf_path)


if __name__ == "__main__":
    main()
