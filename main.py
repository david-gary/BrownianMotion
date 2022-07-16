import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import random

# Brownian Motion Generation Function
np.random.seed(np.random.randint(99999))

dims, N, M, T = 1, 501, 10, 1
t = np.linspace(0, T, N)
dt = T/(N-1)


def generateBrownianMotion(dt, N, dims):
    # Xt ~ N(0, t), t goes from 0 to N
    dX = np.sqrt(dt) * np.random.randn(N)
    X = np.cumsum(dX)

    if dims == 1:
        return (X)

    elif dims == 2:
        dY = np.sqrt(dt) * np.random.randn(N)
        Y = np.cumsum(dY)
        return (X, Y)

    else:
        return None


def animate(num, lines, X1, Y1, X2, Y2):

    lines[0] = animate1D(num, lines[0], X1)
    lines[1] = animate2D(num, lines[1], X1, Y1)
    lines[2] = animate1D(num, lines[2], X2)
    lines[3] = animate2D(num, lines[3], X2, Y2)

    return lines


def animate1D(num, lines, X):

    for line, data in zip(lines, X):
        line.set_data(t[:num], data[:num])
    return lines


def animate2D(num, lines, X, Y):
    for line, dataX, dataY in zip(lines, X, Y):
        line.set_data(dataX[:num], dataY[:num])
    return lines


def make_figure_frames():
    # create a 2x2 grid of subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('X(t)')
    ax1.set_title('Brownian Motion in 1D - Run A')

    ax2.set_xlabel('X(t)')
    ax2.set_ylabel('Y(t)')
    ax2.set_title('2D Brownian Path - Run A')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)

    ax3.set_xlabel('Time t')
    ax3.set_ylabel('X(t)')
    ax3.set_title('Brownian Motion in 1D - Run B')

    ax4.set_xlabel('X(t)')
    ax4.set_ylabel('Y(t)')
    ax4.set_title('2D Brownian Path - Run B')
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-2, 2)

    return fig, ((ax1, ax2), (ax3, ax4))


def cummulative(M, N, dt):
    dX = np.sqrt(dt) * np.random.randn(M, N)
    X = np.cumsum(dX, axis=1)

    dY = np.sqrt(dt) * np.random.randn(M, N)
    Y = np.cumsum(dY, axis=1)
    return X, Y


def main():
    lines = []

    fig, ((ax1, ax2), (ax3, ax4)) = make_figure_frames()

    # Run A
    X = np.array([generateBrownianMotion(dt, N, dims) for i in range(M)])

    lines.append([ax1.plot(t, X[i, :])[0] for i in range(M)])

    X1, Y1 = cummulative(M, N, dt)

    lines.append([ax2.plot(X[i, :], Y1[i, :])[0] for i in range(M)])

    # Run B
    X = np.array([generateBrownianMotion(dt, N, dims) for index in range(M)])

    lines.append([ax3.plot(t, X[i, :])[0] for i in range(M)])

    X2, Y2 = cummulative(M, N, dt)

    lines.append([ax4.plot(X[i, :], Y2[i, :])[0] for i in range(M)])

    # tidy up the figure
    plt.tight_layout()

    ani = animation.FuncAnimation(fig, animate, frames=N,
                                  fargs=(lines, X1, Y1, X2, Y2), interval=15, repeat=True, blit=False)
    plt.show()


if __name__ == '__main__':
    main()
