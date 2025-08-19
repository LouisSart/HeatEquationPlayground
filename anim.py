import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.animation as animation

def animate(T, frames, dt=0.1):
    fig, ax = plt.subplots()
    temp_profile = ax.plot(T[0])[0]

    def update(frame):
        temp_profile.set_ydata(T[frame])
        return temp_profile

    return animation.FuncAnimation(fig, update, frames=frames, interval=dt)