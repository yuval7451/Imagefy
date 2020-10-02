import numpy as np
from integration.utils.visualization import Visualize3D

def main():
    pos = np.random.randint(-10,10,size=(1000,3))
    pos[:,2] = np.abs(pos[:,2])
    vis = Visualize3D(data=pos, save=False)
    # suit.data = pos
    vis.graph()

if __name__ == '__main__':
    main()