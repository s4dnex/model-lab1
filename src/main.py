import math

import numpy as np
import matplotlib.pyplot as plt

with open('../data/numbers.txt', 'r') as f:
    data = np.array([float(line.strip()) for line in f if line.strip()])

plt.figure(figsize=(10, 4))
plt.plot(data, marker='.', linestyle='-', alpha=0.7)
plt.xlabel("Индекс")
plt.ylabel("Значение")
plt.grid(True)
plt.tight_layout()
plt.savefig('../img/plot.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 4))
plt.hist(data, bins=int(math.sqrt(len(data))), edgecolor='black', alpha=0.7)
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.grid(True)
plt.tight_layout()
plt.savefig('../img/hist.png', bbox_inches='tight')
plt.close()

shifts = np.arange(1, 11)
autocorr = [np.corrcoef(data[:-i], data[i:])[0, 1] for i in shifts]
plt.figure(figsize=(10, 4))
plt.plot(shifts, autocorr, marker='.', linestyle='-')
plt.xlabel("Сдвиг")
plt.ylabel("Коэффициент корреляции")
plt.xticks(shifts)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('../img/autocorr.png', bbox_inches='tight')
plt.close()

mean = np.mean(data)
std = np.std(data, ddof=1)
v = std / mean

print(f"Математическое ожидание: {mean:.2f}")
print(f"СКО: {std:.2f}")
print(f"Коэффициент вариации v: {v:.2f}")

q = 1 / (1 + v**2)
t1 = mean * (1 + np.sqrt(((1 - q) / (2 * q)) * (v ** 2 - 1)))
t2 = mean * (1 - np.sqrt((q / (2 * (1 - q))) * (v ** 2 - 1)))

# Генерация
np.random.seed(67)
gen_data = []
n = len(data)
for _ in range(n):
    if np.random.rand() < q:
        gen_data.append(np.random.exponential(t1))
    else:
        gen_data.append(np.random.exponential(t2))
gen_data = np.array(gen_data)

plt.figure(figsize=(10, 4))
plt.hist(data, bins=int(math.sqrt(len(data))), density=True, alpha=0.5, label='Исходная')
plt.hist(gen_data, bins=int(math.sqrt(len(data))), density=True, alpha=0.5, label='Сгенерированная')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../img/hist_compare.png', bbox_inches='tight')
plt.close()

print("(" + ", ".join(f"{x:.2f}" for x in gen_data) + ")")