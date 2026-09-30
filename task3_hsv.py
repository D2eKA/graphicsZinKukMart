# Участник 3: ФИО
# Задание 3. RGB -> HSV, ползунки оттенка, насыщенности и яркости, обратно в RGB.

import sys
from pathlib import Path

import cv2
import numpy as np


def rgb_to_hsv(rgb):
    # rgb: float 0..1, результат H 0..360, S и V 0..1
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask = delta != 0
    red = mask & (cmax == r)
    green = mask & (cmax == g)
    blue = mask & (cmax == b)
    h[red] = 60.0 * (((g[red] - b[red]) / delta[red]) % 6.0)
    h[green] = 60.0 * (((b[green] - r[green]) / delta[green]) + 2.0)
    h[blue] = 60.0 * (((r[blue] - g[blue]) / delta[blue]) + 4.0)
    h = np.mod(h, 360.0)

    s = np.zeros_like(cmax)
    nonzero = cmax != 0
    s[nonzero] = delta[nonzero] / cmax[nonzero]
    v = cmax
    return np.dstack((h, s, v))


def hsv_to_rgb(hsv):
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    c = v * s
    hp = h / 60.0
    x = c * (1.0 - np.abs(np.mod(hp, 2.0) - 1.0))
    m = v - c

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    sectors = [
        ((0 <= hp) & (hp < 1), c, x, 0),
        ((1 <= hp) & (hp < 2), x, c, 0),
        ((2 <= hp) & (hp < 3), 0, c, x),
        ((3 <= hp) & (hp < 4), 0, x, c),
        ((4 <= hp) & (hp < 5), x, 0, c),
        ((5 <= hp) & (hp < 6), c, 0, x),
    ]
    for mask, rr, gg, bb in sectors:
        r[mask] = rr if np.isscalar(rr) else rr[mask]
        g[mask] = gg if np.isscalar(gg) else gg[mask]
        b[mask] = bb if np.isscalar(bb) else bb[mask]

    rgb = np.dstack((r + m, g + m, b + m))
    return np.clip(rgb, 0.0, 1.0)


def apply_sliders(hsv, hue_shift, sat_percent, val_percent):
    changed = hsv.copy()
    changed[:, :, 0] = np.mod(changed[:, :, 0] + hue_shift, 360.0)
    changed[:, :, 1] = np.clip(changed[:, :, 1] * (sat_percent / 100.0), 0.0, 1.0)
    changed[:, :, 2] = np.clip(changed[:, :, 2] * (val_percent / 100.0), 0.0, 1.0)
    return changed


def main():
    if len(sys.argv) < 2:
        print("Запуск: python task3_hsv.py изображение.png")
        print("Ползунки: оттенок, насыщенность, яркость. S — сохранить, Esc — выход.")
        return

    src = Path(sys.argv[1])
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        print("Не удалось открыть", src)
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    base_hsv = rgb_to_hsv(rgb)
    out_path = src.parent / "hsv_result.png"

    window = "HSV"
    cv2.namedWindow(window)
    cv2.createTrackbar("Hue +", window, 0, 360, lambda v: None)
    cv2.createTrackbar("Sat %", window, 100, 200, lambda v: None)
    cv2.createTrackbar("Val %", window, 100, 200, lambda v: None)

    print("S — сохранить hsv_result.png, Esc — закрыть окно.")
    while True:
        hue_shift = cv2.getTrackbarPos("Hue +", window)
        sat_percent = cv2.getTrackbarPos("Sat %", window)
        val_percent = cv2.getTrackbarPos("Val %", window)
        edited = apply_sliders(base_hsv, hue_shift, sat_percent, val_percent)
        result = hsv_to_rgb(edited)
        view = cv2.cvtColor((result * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow(window, view)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        if key in (ord("s"), ord("S")):
            cv2.imwrite(str(out_path), view)
            print("Сохранено:", out_path)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
