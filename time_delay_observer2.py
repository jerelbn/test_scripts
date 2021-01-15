# Example of a simple time delay observer
import matplotlib.pyplot as plt
from numpy import array, sin, cos, tan, vectorize, NaN, clip, argmax, pi, zeros, eye, arccos, trace, mod, vstack, reshape
from numpy.random import rand, randn, seed
from numpy.linalg import qr
from scipy.linalg import circulant, norm

cam_dt = 30 # ms

def Omega(t):
    freq = 1.0 * 2 * pi
    return array([0.3*cos(freq*t)*sin(t), 0.3*sin(freq*t)*cos(t), 0.01*sin(t)])

def skew(v):
    return array([[  0.0, -v[2],  v[1]],
                  [ v[2],   0.0, -v[0]],
                  [-v[1],  v[0],  0.0]])

def vex(V):
    return array([V[2,1], V[0,2], V[1,0]])

# delta = 3x1 vector
def expR(delta):
    theta = norm(delta)
    deltax = skew(delta)
    if (theta > 1e-6):
        return eye(3) + sin(theta) / theta * deltax + (1 - cos(theta)) / theta / theta * deltax * deltax
    else:
        return eye(3)

def logR(R):
    # rotation magnitude
    theta = arccos((trace(R)-1.0)/2.0)

    # avoid numerical error with approximation
    if (theta > 1e-6):
        deltax = theta/(2.0*sin(theta))*(R - R.T)
    else:
        deltax = 0.5*(R - R.T)

    return deltax

def integrateGyro(t_hist, omega_hist, td):
    # timestamps of second and first images
    t2 = t_hist[-1] - td
    t1 = t_hist[-1] - td - cam_dt/1000.0

    # find index of time at second image
    idx1 = -1
    while t_hist[idx1] >= (t2):
        idx1 -= 1
    
    # find index of time at first image
    idx2 = -1
    while t_hist[idx2] >= (t1):
        idx2 -= 1
    
    # integrate gyro between images
    R = eye(3)
    while idx2 < idx1:
        idx2 += 1
        dt = t_hist[idx2] - t_hist[idx2-1]
        omega = 0.5*(omega_hist[idx2] + omega_hist[idx2-1]) # trapezoidal integration
        R = expR(-omega*dt).dot(R)
    
    return R

class Observer:
    def __init__(self, td):
        self.td = td

        self.k_td = 5.0e1
        self.eps = 0.005

    def update(self, t_hist, omega_hist, dR_meas):

        # integrate gyro to obtain dR
        dR = integrateGyro(t_hist, omega_hist, self.td)
        
        # derivatives needed to obtain td
        tdp = self.td + self.eps
        tdm = self.td - self.eps
        dRp = norm(vex(integrateGyro(t_hist, omega_hist, tdp)))
        dRm = norm(vex(integrateGyro(t_hist, omega_hist, tdm)))
        ddR_dtd = (dRp - dRm)/(tdp - tdm)
        
        # error terms
        dR_err = norm(vex(logR(dR_meas.dot(dR.T))))
        td_err = -self.k_td*(ddR_dtd*dR_err)

        # update time delay
        self.td += td_err
        self.td = clip(self.td, 0.0, 1.0)

def main():
    # RNG params
    # seed(0)

    # Simulation setup
    t0 = 0
    tf = 10000
    dt = 2

    td = 200

    sigma_omega = 1e-6

    t_meas_stop = 40000
    td_idx = int(td/dt)
    tdc_idx = int((td+cam_dt)/dt)

    observer = Observer(0.01)

    # Containers
    t_hist = []
    td_hist = []
    td_hat = []
    omega_hist = []
    R_hist = []

    # Step through time
    t = t0
    R = eye(3)
    while (t <= tf):
        # Print and store things
        print("%d %d %d" % (t, td, int(observer.td*1000)))
        t_hist += [t/1000.0]
        td_hist += [td]
        td_hat += [int(observer.td*1000)]
        
        # True kinematics
        if t < t_meas_stop:
            omega = Omega(t/1000.0) + sigma_omega*randn(3)
        else:
            omega = sigma_omega*randn(3)
        omega_hist += [omega]
        R_hist += [R]

        # Update observer
        if t > 1000 and mod(t,30) == 0:
            # Start observer at delayed measurement propagated to current time by delay estimate
            R2 = R_hist[-td_idx]
            R1 = R_hist[-tdc_idx]
            R_meas = R1.dot(R2.T)

            # Run observers
            observer.update(t_hist, omega_hist, R_meas)

        # Increment time
        t += dt

        # Update rotation
        R = expR(omega*dt/1000.0).dot(R)
        R,r = qr(R)

    # Plot results
    fig, axs = plt.subplots(figsize=(18, 10), nrows=3, ncols=1)
    fig.set_facecolor('white')
    vectorize(lambda ax: ax.grid(True))(axs) # add grid to all subplots

    axs[0].set_ylabel('$\\alpha$')
    axs[1].set_ylabel('$\\beta$')
    axs[2].set_ylabel('$t_d$')
    axs[2].set_xlabel('Time (s)')

    # axs[0].plot(t_hist, az_hist, 'b-')
    # axs[1].plot(t_hist, el_hist, 'b-')
    axs[2].plot(t_hist, td_hist, 'b-')

    # axs[0].plot(t_hist, az_hat, 'r--')
    # axs[1].plot(t_hist, el_hat, 'r--')
    axs[2].plot(t_hist, td_hat, 'r--')

    omega_plot = vstack(omega_hist)
    axs[0].plot(t_hist, omega_plot[:,0]*180/pi, 'g-.')
    axs[1].plot(t_hist, omega_plot[:,1]*180/pi, 'g-.')

    plt.show()

if __name__ == "__main__":
    main()