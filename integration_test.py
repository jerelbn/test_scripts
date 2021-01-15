# Compare integration methods
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 0.23 * x**2. + 1.58 * np.sin(x)

def fdot(x):
    return 2. * 0.23 * x + 1.58 * np.cos(x)



# Calculate values
N = 20
x_min = -10.
x_max = 10.
x = np.linspace(x_min, x_max, N+1)
fx = f(x)
fdotx = fdot(x)

# Euler integration
fx_euler = np.full_like(fx, np.NaN)
fx_euler[0] = fx[0]
for i in range(1,N):
    fx_euler[i] = fx_euler[i-1] + fdot(x[i-1]) * (x[i] - x[i-1])

# Trapezoidal integration
fx_trap = np.full_like(fx, np.NaN)
fx_trap[0] = fx[0]
for i in range(1,N+1):
    fx_trap[i] = fx_trap[i-1] + 0.5 * (fdot(x[i-1])+fdot(x[i])) * (x[i]-x[i-1])

# Plot all the things
plt.figure()
plt.plot(x, fx, label='Truth')
plt.plot(x, fx_euler, linestyle='--', label='Euler')
plt.plot(x, fx_trap, linestyle='--', label='Trapezoidal')
plt.xlabel('x')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()
plt.show()
