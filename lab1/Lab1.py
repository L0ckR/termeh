import math
import sympy as s
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


t = s.Symbol('t')

r = 2 + s.sin(6 * t)
fi = 5 * t + 0.2 * s.cos(6 * t)

x = r * s.cos(fi)
y = r * s.sin(fi)

Vx = s.diff(x)
Vy = s.diff(y)

Ax = s.diff(Vx)
Ay = s.diff(Vy)

RKx = -Vy * (Vx ** 2 + Vy ** 2) / (Vx * Ay - Ax * Vy) 
RKy = Vx * (Vx ** 2 + Vy ** 2) / (Vx * Ay - Ax * Vy) 


step = 3000

T = np.linspace(0, 10, step)
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)
RKX = np.zeros_like(T)
RKY = np.zeros_like(T)


for i in np.arange(len(T)):
    X[i] = s.Subs(x, t, T[i])
    Y[i] = s.Subs(y, t, T[i])
    VX[i] = s.Subs(Vx, t, T[i])
    VY[i] = s.Subs(Vy, t, T[i])
    AX[i] = s.Subs(Ax, t, T[i])
    AY[i] = s.Subs(Ay, t, T[i])
    RKX[i] = s.Subs(RKx, t, T[i])
    RKY[i] = s.Subs(RKy, t, T[i])



fig = plt.figure()
grf = fig.add_subplot(1, 1, 1)
grf.axis('equal')
grf.set(xlim = [-5, 5], ylim = [-5, 5])
grf.plot(X,Y)

Pnt = grf.plot(X[0], Y[0], marker = 'o')[0]
Vpl = grf.plot([X[0], X[0]+VX[0]],[Y[0], Y[0]+VY[0]], 'green')[0]
Rvec = grf.plot([0, X[0]],[0, Y[0]], 'red')[0]
Apl = grf.plot([X[0], X[0] + AX[0]], [Y[0], Y[0] + AY[0]], 'blue')[0]
RKpl = grf.plot([X[0], X[0] + RKX[0]], [Y[0], Y[0] + RKY[0]], 'purple')[0]



def Vect_arrow(VecX, VecY, X, Y):
    a = 0.3
    b = 0.2
    Arrx = np.array([-a, 0, -a])
    Arry = np.array([b, 0, -b])

    phi = math.atan2(VecY, VecX)

    RotX = Arrx * np.cos(phi) - Arry * np.sin(phi)
    RotY = Arrx * np.sin(phi) + Arry * np.cos(phi)

    Arrx = RotX + X + VecX
    Arry = RotY + Y + VecY
    
    return Arrx, Arry

ArVX, ArVY = Vect_arrow(VX[0], VY[0], X[0], Y[0])
Varr = grf.plot(ArVX, ArVY, 'green')[0]

ArRX, ArRY = Vect_arrow(X[0], Y[0], 0, 0)
Rarr = grf.plot(ArRX, ArRY, 'red')[0]

ArAX, ArAY = Vect_arrow(AX[0], AY[0], X[0], Y[0])
Aarr = grf.plot(ArAX, ArAY, 'blue')[0]

ArRKX, ArRKY = Vect_arrow(RKX[0], RKY[0], X[0], Y[0])
RKarr = grf.plot(ArRKX, ArRY, 'purple')[0]

def anim(i):
    Pnt.set_data(X[i], Y[i])
    
    Vpl.set_data([X[i], X[i]+VX[i]],[Y[i], Y[i]+VY[i]])
    ArVX, ArVY = Vect_arrow(VX[i], VY[i], X[i], Y[i])
    Varr.set_data(ArVX, ArVY)
    
    Rvec.set_data([0, X[i]], [0, Y[i]])
    ArRX, ArRY = Vect_arrow(X[i], Y[i], 0, 0)
    Rarr.set_data(ArRX, ArRY)

    Apl.set_data([X[i], X[i]+AX[i]],[Y[i], Y[i]+AY[i]])
    ArAX, ArAY = Vect_arrow(AX[i], AY[i], X[i], Y[i])
    Aarr.set_data(ArAX, ArAY)

    RKpl.set_data([X[i], X[i]+RKX[i]],[Y[i], Y[i]+RKY[i]])
    ArRKX, ArRKY = Vect_arrow(RKX[i], RKY[i], X[i], Y[i])
    RKarr.set_data(ArRKX, ArRKY)



animation = FuncAnimation(fig, anim, frames = step, interval = 2)
fig.show()
plt.show()