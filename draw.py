import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
r1=int(input("R1:"))
r2=int(input("R2:"))
f1=np.radians(int(input("fai 1:")))
f2=np.radians(int(input("fai 2:")))
w1=int(input("w1:"))
w2=int(input("w2:"))

t = np.linspace(0,2*np.pi,100)

x = r1 * np.cos(w1 * t + f1)
y = r2 * np.cos(w2 * t + f2)

fig, ax = plt.subplots()
ax.set_xlim(-r1 - 1, r1 + 1)
ax.set_ylim(-r2 - 1, r2 + 1)
line, = ax.plot([], [], 'go')
path, = ax.plot([], [], 'b-')

def init():
    line.set_data([] , [])
    path.set_data([], [])
    return (line, path)

def update(i):
    line.set_data([x[i]],[y[i]])
    path_x = x[:i + 1]
    path_y = y[:i + 1]
    path.set_data(path_x, path_y)
    return (line, path)

ani = FuncAnimation(fig,update,frames=len(t), init_func=init, blit=True)

plt.show()