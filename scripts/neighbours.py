import numpy as np
import matplotlib.pyplot as plt


def show_slice(Z, name, text=""):
    rows,cols = Z.shape
    fig = plt.figure(figsize=(cols/4.,rows/4.), dpi=72)
    #fig.patch.set_alpha(0.0)
    ax = plt.subplot(111)
    A = np.arange(rows*cols).reshape(rows,cols)
    plt.imshow(Z, cmap='Purples', extent=[0,cols,0,rows],
               vmin=0, vmax=1, interpolation='nearest', origin='upper')
    plt.xticks([]), plt.yticks([])
    for pos in ['top', 'bottom', 'right', 'left']:
        ax.spines[pos].set_edgecolor('k')
        ax.spines[pos].set_alpha(.25)
    if text:
        ax.text(cols/2.,-0.15,text,color='k',ha='center',va='top',
                family='monospace', weight='bold',fontsize=10)
    plt.savefig('../figures/%s' % name, dpi=72)
    #plt.show()

rows,cols = 6, 6

Z = np.zeros((rows,cols))+.1
Z[1:-1,1:-1] = .25
Z[ :-2,  :-2] = .5
show_slice(Z, '../figures/neighbours-1.png', "Z[:-2,:-2]")

Z = np.zeros((rows,cols))+.1
Z[1:-1,1:-1] = .25
Z[ :-2, 1:-1] = .5
show_slice(Z, '../figures/neighbours-2.png',"Z[:-2,1:-1]")

Z = np.zeros((rows,cols))+.1
Z[1:-1,1:-1] = .25
Z[ :-2, 2:] = .5
show_slice(Z, '../figures/neighbours-3.png',"Z[:-2,2:]")


Z = np.zeros((rows,cols))+.1
Z[1:-1,1:-1] = .25
Z[1:-1, 0:-2] = .5
show_slice(Z, '../figures/neighbours-4.png', "Z[1:-1,:-2]")

Z = np.zeros((rows,cols))+.1
Z[1:-1,1:-1] = .25
show_slice(Z, '../figures/neighbours-5.png',"Z[1:-1,1:-1]")

Z = np.zeros((rows,cols))+.1
Z[1:-1,1:-1] = .25
Z[1:-1, 2: ] = .5
show_slice(Z, '../figures/neighbours-6.png', "Z[1:-1,2:]")

Z = np.zeros((rows,cols))+.1
Z[1:-1,1:-1] = .25
Z[2:,  :-2] = .5
show_slice(Z, '../figures/neighbours-7.png', "Z[2:,:-2]")

Z = np.zeros((rows,cols))+.1
Z[1:-1,1:-1] = .25
Z[2:, 1:-1] = .5
show_slice(Z, '../figures/neighbours-8.png', "Z[2:,1:-1]")

Z = np.zeros((rows,cols))+.1
Z[1:-1,1:-1] = .25
Z[2:,   2:] = .5
show_slice(Z, '../figures/neighbours-9.png', "Z[2:,2:]")

