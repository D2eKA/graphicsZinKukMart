# Участник 2: ФИО
# Задание 2. Каналы R, G, B и три цветовые гистограммы.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_rgb(path):
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def channel_image(rgb, index):
    # Остальные каналы обнуляются, остаётся только выбранный цвет.
    result = np.zeros_like(rgb)
    result[:, :, index] = rgb[:, :, index]
    return result


def show_result(rgb):
    red = channel_image(rgb, 0)
    green = channel_image(rgb, 1)
    blue = channel_image(rgb, 2)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes[0, 0].imshow(red)
    axes[0, 0].set_title("Канал R")
    axes[0, 1].imshow(green)
    axes[0, 1].set_title("Канал G")
    axes[0, 2].imshow(blue)
    axes[0, 2].set_title("Канал B")

    colors = ("red", "green", "blue")
    titles = ("Гистограмма R", "Гистограмма G", "Гистограмма B")
    for i in range(3):
        axes[1, i].hist(rgb[:, :, i].ravel(), bins=256, range=(0, 255), color=colors[i])
        axes[1, i].set_title(titles[i])
        axes[1, i].set_xlim(0, 255)
        axes[1, i].set_xlabel("Значение канала")
        axes[1, i].set_ylabel("Число пикселей")

    for ax in axes[0]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()
    return red, green, blue


def main():
    if len(sys.argv) < 2:
        print("Запуск: python task2_channels.py изображение.png")
        return

    src = Path(sys.argv[1])
    rgb = load_rgb(src)
    red = channel_image(rgb, 0)
    green = channel_image(rgb, 1)
    blue = channel_image(rgb, 2)
    Image.fromarray(red).save(src.parent / "channel_r.png")
    Image.fromarray(green).save(src.parent / "channel_g.png")
    Image.fromarray(blue).save(src.parent / "channel_b.png")
    print("Сохранено: channel_r.png, channel_g.png, channel_b.png")
    show_result(rgb)


if __name__ == "__main__":
    main()
