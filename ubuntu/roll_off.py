import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('font',family='serif')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

x = np.linspace(0, 1200, 1000)

roll_off = 6e-3
plt.plot(x, np.maximum(1-roll_off*x, 0.), label=r'$r = 6e^{-3}$')

roll_off = 4e-3
plt.plot(x, np.maximum(1-roll_off*x, 0.), label=r'$r = 4e^{-3}$')

roll_off = 2e-3
plt.plot(x, np.maximum(1-roll_off*x, 0.), label=r'$r = 2e^{-3}$')

roll_off = 1e-3
plt.plot(x, np.maximum(1-roll_off*x, 0.), label=r'$r = 1e^{-3}$')

plt.title(r'Confidence')
plt.xlabel(r'Var/Pixel')
plt.ylabel(r'Con/a.u.')
plt.grid()
plt.legend()
plt.savefig('confidence.pdf')
