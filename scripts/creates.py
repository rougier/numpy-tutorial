import numpy as np
import matplotlib.pyplot as plt


def show_array2(Z, name):
    Z = np.atleast_2d(Z)
    rows,cols = Z.shape
    fig = plt.figure(figsize=(cols/4.,rows/4.), dpi=72)
    ax = plt.subplot(111)
    plt.imshow(Z, cmap='Purples', extent=[0,cols,0,rows],
               vmin=0, vmax=max(1,Z.max()), interpolation='nearest',
               origin='upper')
    #plt.xticks(1.05+np.arange(cols-1),[]), plt.yticks(1+np.arange(rows-1),[])
    plt.xticks([]), plt.yticks([])
    for pos in ['top', 'bottom', 'right', 'left']:
        ax.spines[pos].set_edgecolor('k')
        ax.spines[pos].set_alpha(.25)
    plt.savefig('../figures/%s' % name, dpi=72)
    #plt.show()


def show_array3(Z, name):
    Z = np.atleast_2d(Z)
    rows,cols = Z.shape[1],Z.shape[2]
    fig = plt.figure(figsize=(cols,rows), dpi=72, frameon=False)
    for i in range(Z.shape[0],0,-1):
        d = .2*i/float(Z.shape[0])
        ax = plt.axes([d,d,0.7,0.7])
        plt.imshow(Z[Z.shape[0]-i], cmap='Purples', extent=[0,cols,0,rows],
                   vmin=0, vmax=max(1,Z.max()), interpolation='nearest',
                   origin='upper')
        plt.xticks([]), plt.yticks([])
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor('k')
            ax.spines[pos].set_alpha(.25)
    plt.savefig('../figures/%s' % name, dpi=16)
    #plt.show()



rows,cols = 5, 9

Z = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.])
Z += 0.1
show_array2(Z, 'create-list-1.png')

Z = np.zeros(cols)+.1
show_array2(Z, 'create-zeros-1.png')

Z = np.ones(cols)+.1
show_array2(Z, 'create-ones-1.png')

Z = np.arange(cols)
show_array2(Z, 'create-arange-1.png')

Z = np.random.uniform(0,1,cols)
show_array2(Z, 'create-uniform-1.png')

Z = np.zeros((rows,cols))+.1
show_array2(Z, 'create-zeros-2.png')

Z = np.ones((rows,cols))+.1
show_array2(Z, 'create-ones-2.png')

Z = np.array([[0.,0.,0.,0.,0.,0.,0.,0.,0.],
              [0.,0.,0.,0.,0.,0.,0.,0.,0.],
              [0.,0.,0.,0.,0.,0.,0.,0.,0.],
              [0.,0.,0.,0.,0.,0.,0.,0.,0.],
              [0.,0.,0.,0.,0.,0.,0.,0.,0.]])
Z += 0.1
show_array2(Z, 'create-list-2.png')

Z = np.arange(rows*cols).reshape(rows,cols)
show_array2(Z, 'create-arange-2.png')

Z = np.random.uniform(0,1,(rows,cols))
show_array2(Z, 'create-uniform-2.png')

Z = np.zeros((3,5,9))+.1
show_array3(Z, 'create-zeros-3.png')

Z = np.ones((3,5,9))+.1
show_array3(Z, 'create-ones-3.png')

Z = np.arange(3*5*9).reshape(3,5,9)
show_array3(Z, 'create-arange-3.png')

Z = np.random.uniform(0,1,(3,rows,cols))
show_array3(Z, 'create-uniform-3.png')


