import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle


mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command


def load_csv(loc):
    data = np.genfromtxt(loc, delimiter=',', dtype=float)
    return data


def plot_time(locs):
    
    mean = np.empty([len(locs)])
    std = np.empty([len(locs)])

    for i, loc in enumerate(locs):
        data = load_csv('data/' + loc)
        mean[i] = data.mean()
        std[i] = data.std()

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.errorbar(mean[2], 2, xerr=std[2], fmt='|')
    ax2.errorbar(mean[1], 1, xerr=std[1], fmt='|')
    ax3.errorbar(mean[0], 0, xerr=std[0], fmt='|')

    ax1.set_xlim(40, 60)
    ax2.set_xlim(145, 150)
    ax3.set_xlim(840, 900)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    ax1.yaxis.tick_left()
    ax2.get_yaxis().set_ticks([])
    ax3.yaxis.tick_right()

    # add diagonal lines
    d = 0.02
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch axes
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)        
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax3.transAxes)  # switch axes 
    ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax3.plot((-d, +d), (-d, +d), **kwargs)

    plt.show()

def plot_3d(loc):
    data = load_csv(loc)

    a = 0
    z = data.shape[0]
    com_x = data[a:z, 0]
    com_y = data[a:z, 3]
    com_z = data[a:z, 6]
    left_x = data[a:z, 13]
    left_y = data[a:z, 14]
    left_z = data[a:z, 15]
    right_x = data[a:z, 17]
    right_y = data[a:z, 18]
    right_z = data[a:z, 19]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(com_x, com_y, com_z, label='CoM')
    ax.plot(left_x, left_y, left_z, label='Left Foot')
    ax.plot(right_x, right_y, right_z, label='Right Foot')

    ax.set_xlabel(r'$\text{x}\,/\,\text{m}$')
    ax.set_ylabel(r'$\text{y}\,/\,\text{m}$')
    ax.set_zlabel(r'$\text{z}\,/\,\text{m}$')	

    ax.view_init(45, -115) 

    plt.legend()
    #plt.title('Walking Trajectories')
    #plt.show()

    out = ''

    if (loc[0] == 'p'):
        out = 'img/{}{}'.format(loc[7:-4], '.pdf')
    else:
        out = 'img/{}{}'.format(loc[4:-4], '.pdf')

    plt.savefig(out)


def plot_with_obstacle(loc):
    data = load_csv(loc)

    a = 0
    z = data.shape[0]
    com_x = data[a:z, 0]
    com_y = data[a:z, 3]
    com_z = data[a:z, 6]
    left_x = data[a:z, 13]
    left_y = data[a:z, 14]
    left_z = data[a:z, 15]
    right_x = data[a:z, 17]
    right_y = data[a:z, 18]
    right_z = data[a:z, 19]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(com_x, com_y, label='CoM')
    ax.plot(left_x, left_y, label='Left Foot')
    ax.plot(right_x, right_y, label='Right Foot')

    ax.set_xlabel(r'$\text{x}\,/\,\text{m}$')
    ax.set_ylabel(r'$\text{y}\,/\,\text{m}$')

    # circle with radius = radius + margin + max foot
    circle = plt.Circle(([1.6, 1.0]), 1.0, linestyle='-', color='gray', fill=False, label='Obstacle')
    ax.add_patch(circle)

    circle = plt.Circle(([1.6, 1.0]), 1.0 + 0.2 + 0.2172, linestyle='--', color='gray', fill=False, label='Security Margin')
    ax.add_patch(circle)

    ax.set_xlim( 0.0, 4.5)
    ax.set_ylim(-1.0, 2.5)

    plt.legend()
    #plt.title('Walking Trajectories')
    #plt.show()

    out = ''

    if (loc[0] == 'p'):
        out = 'img/{}{}'.format(loc[7:-4], '.pdf')
    else:
        out = 'img/{}{}'.format(loc[4:-4], '.pdf')

    plt.savefig(out)


def plot_dif(loc):
    data = load_csv(loc)

    a = 0
    z = data.shape[0]

    dx = data[a:z, 0]
    dy = data[a:z, 1]
    dz = data[a:z, 2]

    time = np.linspace(0, dx.shape[0]*0.01, dx.shape[0]) 

    plt.plot(time, dx, label='dx')
    plt.plot(time, dy, label='dy')
    plt.plot(time, dz, label='dz')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Evaluate time.
    time_locs = ['python_nmpc_time.csv',
                 'cpp_nmpc_time.csv',
                 'cpp_mpc_time.csv']

    # use a table instead! errors scale not well
    # plot_time(time_locs)

    # Plot 3d data without obstacle.
    data_locs = ['python_nmpc_straight.csv',
                 'python_nmpc_diagonal.csv',
                 'python_nmpc_turn.csv',
                 'cpp_nmpc_straight.csv',
                 'cpp_nmpc_diagonal.csv',
                 'cpp_nmpc_turn.csv',
                 'cpp_mpc_straight.csv',
                 'cpp_mpc_diagonal.csv',
                 'cpp_mpc_turn.csv']

    for loc in data_locs:
        plot_3d('data/' + loc)

    # Plot 2d data with obstacle.
    data_locs = ['python_nmpc_obstacle.csv',
                 'cpp_nmpc_obstacle.csv']

    for loc in data_locs:
        plot_with_obstacle('data/' + loc)
