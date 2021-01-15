# Example of a simple time delay observer
import matplotlib.pyplot as plt
from numpy import sin, cos, tan, vectorize, NaN, clip, argmax, pi
from numpy.random import rand, randn, seed
from scipy.linalg import circulant

def propMeasForward(t_hist, wx_hist, wy_hist, wz_hist, az, el, td):
    # push meas forward to current time
    idx = -1
    while t_hist[idx] >= (t_hist[-1] - td):
        idx -= 1
    
    while idx < -1:
        idx += 1
        dt = t_hist[idx] - t_hist[idx-1]
        tan_az = tan(az)
        tan_el = tan(el)
        az_dot = -((wx_hist[idx]*tan_az + wz_hist[idx])*tan_el)/(tan_az**2.0 + 1.0) - wy_hist[idx]
        el_dot = -((wy_hist[idx]*tan_el - wz_hist[idx])*tan_az)/(tan_el**2.0 + 1.0) - wx_hist[idx]
        az += az_dot*dt
        el += el_dot*dt
    
    return az, el

class Observer:
    def __init__(self, az, el, td):
        self.az = az
        self.el = el
        self.td = td

        self.k_az = 3.0
        self.k_el = 3.0
        self.k_td = 3.0

    def update(self, t_hist, az_meas, el_meas, wx_hist, wy_hist, wz_hist, dt):
        
        # push meas forward to current time
        az, el = propMeasForward(t_hist, wx_hist, wy_hist, wz_hist, az_meas, el_meas, self.td)
        
        # derivatives needed to obtain td
        tdp = self.td + dt
        tdm = self.td - dt
        azp, elp = propMeasForward(t_hist, wx_hist, wy_hist, wz_hist, az_meas, el_meas, tdp)
        azm, elm = propMeasForward(t_hist, wx_hist, wy_hist, wz_hist, az_meas, el_meas, tdm)
        daz_dtd = (azp - azm)/(tdp - tdm)
        del_dtd = (elp - elm)/(tdp - tdm)
        
        # error terms
        az_err = self.k_az*(az - self.az)
        el_err = self.k_el*(el - self.el)
        td_err = -self.k_td*(daz_dtd*az_err + del_dtd*el_err)

        # derivatives of current state with added correction error terms
        tan_az = tan(self.az)
        tan_el = tan(self.el)
        az_dot = -((wx_hist[-1]*tan_az + wz_hist[-1])*tan_el)/(tan_az*tan_az + 1.0) - wy_hist[-1] + az_err
        el_dot = -((wy_hist[-1]*tan_el - wz_hist[-1])*tan_az)/(tan_el*tan_el + 1.0) - wx_hist[-1] + el_err
        td_dot = td_err

        # Euler integration
        self.az += az_dot*dt
        self.el += el_dot*dt
        self.td += td_dot*dt
        self.td = clip(self.td, 0.0, 1.0)

class Observer2:
    def __init__(self):
        self.t_prev = 0.0
        self.az_prev = 0.0
        self.el_prev = 0.0
        self.td = 0.0

    def update(self, t_hist, az, el, wx_hist, wy_hist, wz_hist):
        t = t_hist[-1]
        
        if az == az and el == el:
            # numerical derivative of az/al
            az_dot_num = (az - self.az_prev) / (t - self.t_prev)
            el_dot_num = (el - self.el_prev) / (t - self.t_prev)

            # search analytical solutions as function of angular rate for closest solution to time delay
            idx = -1
            err_prev = 999
            for i in range(-1,-100,-1):
                wx = wx_hist[i]
                wy = wy_hist[i]
                wz = wz_hist[i]
                tan_az = tan(az)
                tan_el = tan(el)
                az_dot_anal = -((wx*tan_az + wz)*tan_el)/(tan_az*tan_az + 1.0) - wy
                el_dot_anal = -((wy*tan_el - wz)*tan_az)/(tan_el*tan_el + 1.0) - wx

                err = (az_dot_num - az_dot_anal)**2.0 + (el_dot_num - el_dot_anal)**2.0

                if err < err_prev:
                    idx = i
                    err_prev = err

            self.td = t_hist[-1] - t_hist[idx]
        
        self.t_prev = t
        self.az_prev = az
        self.el_prev = el

def main():
    # RNG params
    # seed(0)

    # Simulation setup
    t0 = 0.0
    tf = 10.0
    dt = 0.0025

    az = 0.1
    el = 0.1
    td = 0.16

    sigma_omega = 0.005
    sigma_azel = 1e-4

    freq = 1.0 * 2 * pi
    td_err0 = 0.3
    t_meas_stop = 999.0

    td_idx = int(td/dt)

    observer = Observer(0, 0, td+0.03*randn())
    observer2 = Observer2()

    # Containers
    t_hist = []

    az_hist = []
    el_hist = []
    td_hist = []

    az_hat = []
    el_hat = []
    td_hat = []

    td_hat2 = []

    wx_hist = []
    wy_hist = []
    wz_hist = []

    az_meas_hist = []
    el_meas_hist = []

    daz_hist = []
    del_hist = []

    # Step through time
    t = t0
    first_meas = True
    while (t <= tf):
        # Print and store things
        print("%8.3f %8.3f %8.3f %8.3f" % (t, td, observer.td, observer2.td))
        t_hist += [t]
        az_hist += [az]
        el_hist += [el]
        td_hist += [td]
        az_hat += [observer.az]
        el_hat += [observer.el]
        td_hat += [observer.td]
        td_hat2 += [observer2.td]
        
        # True kinematics
        if t < t_meas_stop:
            omega_x = 0.3*cos(freq*t) + sigma_omega*randn()
            omega_y = 0.3*sin(freq*t) + sigma_omega*randn()
            omega_z = 0.01*sin(t) + sigma_omega*randn()
        else:
            omega_x = sigma_omega*randn()
            omega_y = sigma_omega*randn()
            omega_z = sigma_omega*randn()

        wx_hist += [omega_x]
        wy_hist += [omega_y]
        wz_hist += [omega_z]

        # Update observer
        if t > 0.5:
            az_meas = az_hist[-td_idx] + sigma_azel*randn()
            el_meas = el_hist[-td_idx] + sigma_azel*randn()

            # Start observer at delayed measurement propagated to current time by delay estimate
            if first_meas:
                observer.az, observer.el = propMeasForward(t_hist, wx_hist, wy_hist, wz_hist, az_meas, el_meas, observer.td)
                first_meas = False

            # Run observers
            observer.update(t_hist, az_meas, el_meas, wx_hist, wy_hist, wz_hist, dt)
            observer2.update(t_hist, az_meas, el_meas, wx_hist, wy_hist, wz_hist)
        else:
            az_meas = NaN
            el_meas = NaN
        
        az_meas_hist += [az_meas]
        el_meas_hist += [el_meas]

        # Store numerical derivatives of az/el
        if t > t0:
            daz_hist += [-(az_meas_hist[-1]-az_meas_hist[-2])/dt]
            del_hist += [-(el_meas_hist[-1]-el_meas_hist[-2])/dt]
        else:
            daz_hist += [NaN]
            del_hist += [NaN]

        # Update truth
        tan_az = tan(az)
        tan_el = tan(el)
        az_dot = -((omega_x*tan_az + omega_z)*tan_el)/(tan_az*tan_az + 1.0) - omega_y
        el_dot = -((omega_y*tan_el - omega_z)*tan_az)/(tan_el*tan_el + 1.0) - omega_x

        az += az_dot*dt
        el += el_dot*dt

        # Increment time
        t += dt

    # Plot results
    fig, axs = plt.subplots(figsize=(18, 10), nrows=3, ncols=1)

    fig.set_facecolor('white')

    vectorize(lambda ax: ax.grid(True))(axs) # add grid to all subplots

    axs[0].set_ylabel('$\\alpha$')
    axs[1].set_ylabel('$\\beta$')
    axs[2].set_ylabel('$t_d$')
    axs[2].set_xlabel('Time (s)')

    axs[0].plot(t_hist, az_hist, 'b-')
    axs[1].plot(t_hist, el_hist, 'b-')
    axs[2].plot(t_hist, td_hist, 'b-')

    axs[0].plot(t_hist, az_hat, 'r--')
    axs[1].plot(t_hist, el_hat, 'r--')
    axs[2].plot(t_hist, td_hat, 'r--')

    # axs[2].plot(t_hist, td_hat2, 'g-.')


    # axs[0].plot(t_hist, wy_hist, 'g-.')
    # axs[1].plot(t_hist, wx_hist, 'g-.')

    # axs[0].plot(t_hist, daz_hist, 'c-.')
    # axs[1].plot(t_hist, del_hist, 'c-.')

    plt.show()

if __name__ == "__main__":
    main()