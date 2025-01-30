import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple


# Na podstawie wartosc r oraz momentu siły m liczymy wartosci B na osiach (B to wektory)
def liczenie_ze_wzoru(m: np.ndarray, r: np.ndarray):
    # Wywalamy wartosci bliskie zeru bo bedziemy przez nie dzielic wiec zeby błędu nie wywalilo
    r = r[np.linalg.norm(r, axis=1) > 1e-6]
    prod = (3 * (r @ m))[:, np.newaxis] * r  # (n, 3) tablica (pierwsza częśc licznika we wzorze liczona)
    r_magnitude = np.linalg.norm(r, axis=1, keepdims=True)  # (n, 1) tablica (liczy dlugość każdego r)

    B = 1e-7 * (prod / np.power(r_magnitude, 5) - m / np.power(r_magnitude, 3))  # wynik końcowy (u0/4pi = 10^-7)
    return r, B  # punkty i tablica(B) z ich kordami


# Przedziały dla osi
axis = (np.s_[-1.8:1:.5], np.s_[-1.8:1:.5], np.s_[-1.8:1:.5])  # rozmiar wykresu
r = np.asarray([a.ravel() for a in np.mgrid[axis]]).T

# Wyniki w macierzach
m = np.array([0, 0, 1])  # moment magnetyczny
r, B = liczenie_ze_wzoru(m, r)
print("wartości r")
print(r)
print("odpowiadające im kordy")
print(B)

# Wykres
x, y, z = r[:, 0], r[:, 1], r[:, 2]
u, v, w = B[:, 0], B[:, 1], B[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=90, azim=90)  # Początkowy kąt widoku (można zmienić)
ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
plt.show()