import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
mpl.rcParams.update({'font.size': 20})


def load_csv(loc):
    data = np.genfromtxt(loc, delimiter=',', dtype=float)
    return data


data = load_csv('loss_hist.csv')
epoch = data[:,0]
reward = data[:,1]
std = data[:,2]

plt.plot(epoch, reward, label='Total Reward')
plt.xlabel(r'\text{Epoch}\,[\#]')
plt.ylabel(r'\text{Total Reward}\,[\text{a.u.}]')
plt.title('Reward History')
plt.legend(loc='lower right')
plt.grid()
#plt.show()
plt.tight_layout()
plt.savefig('nmpc_ppo_reward_history.pdf')

plt.clf()

plt.plot(epoch, std, label='Standard Deviation', c='C1')
plt.xlabel(r'\text{Epoch}\,[\#]')
plt.ylabel(r'\text{Standard Deviation}\,[\text{m}/\text{s}]')
plt.title('Standard Deviation History')
plt.legend()
plt.grid()
#plt.show()
plt.tight_layout()
plt.savefig('nmpc_ppo_std_history.pdf')
