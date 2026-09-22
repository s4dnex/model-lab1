import numpy as np
import matplotlib.pyplot as plt

with open('../data/numbers.txt', 'r') as f:
    data = np.array([float(line.strip()) for line in f if line.strip()])

plt.figure(figsize=(10, 4))
plt.plot(data, marker='.', linestyle='-', alpha=0.7)
plt.title("График значений числовой последовательности")
plt.xlabel("Индекс")
plt.ylabel("Значение")
plt.grid(True)
plt.savefig('../img/plot.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 4))
plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.title("Гистограмма распределения частот")
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.grid(True)
plt.savefig('../img/hist.png', bbox_inches='tight')
plt.close()

shifts = np.arange(1, 11)
autocorr = [np.corrcoef(data[:-i], data[i:])[0, 1] for i in shifts]
plt.figure(figsize=(10, 4))
plt.bar(shifts, autocorr, color='skyblue', edgecolor='black')
plt.title("Автокорреляционная функция")
plt.xlabel("Сдвиг")
plt.ylabel("Коэффициент корреляции")
plt.xticks(shifts)
plt.grid(axis='y')
plt.savefig('../img/autocorr.png', bbox_inches='tight')
plt.close()

mean_val = np.mean(data)
std_val = np.std(data, ddof=1)
v = std_val / mean_val

print(f"Математическое ожидание: {mean_val:.2f}")
print(f"СКО: {std_val:.2f}")
print(f"Коэффициент вариации v: {v:.2f}")

q = 1 / (1 + v**2)
t1 = mean_val * (1 + np.sqrt(((1 - q) / (2 * q)) * (v**2 - 1)))
t2 = mean_val * (1 - np.sqrt((q / (2 * (1 - q))) * (v**2 - 1)))

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
plt.hist(data, bins=30, density=True, alpha=0.5, label='Исходная')
plt.hist(gen_data, bins=30, density=True, alpha=0.5, label='Сгенерированная')
plt.title("Сравнение гистограмм")
plt.legend()
plt.grid(True)
plt.savefig('../img/hist_compare.png', bbox_inches='tight')
plt.close()

print("(" + ", ".join(f"{x:.2f}" for x in gen_data) + ")")