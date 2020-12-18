import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

X = np.linspace(-5.12, 5.12, 100)
Y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(X, Y)

Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + \
    (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap=cm.nipy_spectral, linewidth=0.08,
                antialiased=True)
# plt.savefig('rastrigin_graph.png')
plt.show()
