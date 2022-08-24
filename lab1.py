import warnings

import cv2 as cv
import numpy as np
import random
from pathlib import Path
import collections
from matplotlib import pyplot as plt
from scipy.stats._continuous_distns import _distn_names
import scipy.stats as st
import pandas as pd


def make_pdf(dist, params, size=10000):
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


def best_fit_distribution(data, bins=200, ax=None):
    global distrs
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    best_distributions = []
    for ii, distribution in enumerate([d for d in distrs if d in _distn_names]):
        distribution = getattr(st, distribution)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                params = distribution.fit(data)
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass
                best_distributions.append((distribution, params, sse))
        except Exception:
            pass

    return sorted(best_distributions, key=lambda x: x[2])[0]


show_plots = True
res_to_file = True
distrs = ["norm", "gamma", "laplace"]
sample_images_list = []
random.seed(8)
for i in range(250):
    num = random.randint(1, 25000)
    sample_images_list.append(str(Path("mirflickr", f"im{num}.jpg")))
output = []
distributions = {}
max_pic_number = 2
for number_pic in range(0, max_pic_number):
    print(f"Picture [{number_pic + 1}/{max_pic_number}]: {sample_images_list[number_pic]}")
    output.append(f"Picture: {sample_images_list[number_pic]}")
    img = cv.imread(sample_images_list[number_pic])
    b, g, r = cv.split(img)
    channels = {
        "r": r,
        "g": g,
        "b": b}

    ttl = img.size / 3  # divide by 3 to get the number of image PIXELS
    for key in channels:
        output.append(f"Для {key} канала:")
        # Среднее значение
        # Максимальное и минимальное значение:
        c_max = np.amax(channels[key])
        c_min = np.amin(channels[key])
        output.append(f"    Max: {c_max}, Min: {c_min}")

        c_matr = channels[key]
        c_2 = [num for numbers in c_matr for num in numbers]
        c_amounts = dict(collections.Counter(c_2))
        M = 0
        for number in c_amounts:
            M += (number * c_amounts[number] / len(c_2))
        output.append(f"    Матожидание: {M}")
        D = 0
        for number in c_amounts:
            D += (number ** 2 * c_amounts[number] / len(c_2))
        D -= M ** 2
        output.append(f"    Дисперсия: {D}")

        output.append(f"    Медиана: {np.median(c_2)}")

        output.append(f"    Интерквартильний розмах: {np.percentile(c_2, 75) - np.percentile(c_2, 25)}")

        # Вероятность появления интенсивности:
        hyst = {}
        all_pix = len(c_2)
        for numm in c_amounts:
            hyst[numm] = c_amounts[numm] / all_pix

        data = pd.Series(channels[key].ravel())

        plt.figure(figsize=(12, 8))
        ax = data.plot(kind='hist', bins=256, density=True, color="red")
        yLim = ax.get_ylim()
        best_distr = best_fit_distribution(data, bins=256, ax=ax)
        pdf = make_pdf(best_distr[0], best_distr[1])
        ax = pdf.plot(label=f"BEST-{best_distr[0].name}", legend=True)
        ax.set_ylim(yLim)
        ax.set_xlim(0)
        data.plot(kind='hist', bins=256, density=True, label="DATA", legend=True)
        plt.title(f"Pic: {sample_images_list[number_pic]}, Channel: {key}")

        if show_plots:
            plt.show()
        else:
            plt.close()
        dist_name = best_distr[0].name
        output.append(f"    Лучшее распределение: {dist_name}")

        if dist_name in distributions:
            distributions[dist_name] += 1
        else:
            distributions[dist_name] = 1
    output.append("========================================")
for line in output:
    print(line)

print(distributions)
if res_to_file:
    with open("results.txt", "w") as file:
        for line in output:
            file.write(line + "\n")
        for k in distributions:
            file.write(f"{k}: {distributions[k]}\n")





