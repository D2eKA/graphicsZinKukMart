# Участник 1: ФИО
# Задание 1. Два перевода RGB в серый, их разность и две гистограммы.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_rgb(path):
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float64)


def to_gray_ntsc(rgb):
    # Вклад каналов по NTSC: больше всего зелёный.
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return 0.30 * r + 0.59 * g + 0.11 * b


def to_gray_srgb(rgb):
    # Другие веса, ближе к sRGB: зелёный ещё сильнее, синий слабее.
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return 0.21 * r + 0.72 * g + 0.078 * b


def show_result(gray_ntsc, gray_srgb, diff):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    axes[0, 0].imshow(gray_ntsc, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Серый, NTSC\n0.30R + 0.59G + 0.11B")
    axes[0, 1].imshow(gray_srgb, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Серый, sRGB\n0.21R + 0.72G + 0.078B")
    axes[0, 2].imshow(diff, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Разность |NTSC - sRGB|")

    axes[1, 0].hist(gray_ntsc.ravel(), bins=256, range=(0, 255), color="gray")
    axes[1, 0].set_title("Гистограмма NTSC")
    axes[1, 1].hist(gray_srgb.ravel(), bins=256, range=(0, 255), color="gray")
    axes[1, 1].set_title("Гистограмма sRGB")
    axes[1, 2].axis("off")

    for ax in axes[0]:
        ax.axis("off")
    for ax in axes[1, :2]:
        ax.set_xlim(0, 255)
        ax.set_xlabel("Интенсивность")
        ax.set_ylabel("Число пикселей")

    fig.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Запуск: python task1_grayscale.py изображение.png")
        return

    src = Path(sys.argv[1])
    out_dir = src.parent
    rgb = load_rgb(src)

    gray_ntsc = to_gray_ntsc(rgb)
    gray_srgb = to_gray_srgb(rgb)
    diff = np.abs(gray_ntsc - gray_srgb)

    Image.fromarray(np.clip(gray_ntsc, 0, 255).astype(np.uint8)).save(out_dir / "gray_ntsc.png")
    Image.fromarray(np.clip(gray_srgb, 0, 255).astype(np.uint8)).save(out_dir / "gray_srgb.png")
    Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8)).save(out_dir / "gray_diff.png")
    print("Сохранено: gray_ntsc.png, gray_srgb.png, gray_diff.png")

    show_result(gray_ntsc, gray_srgb, diff)


if __name__ == "__main__":
    main()
