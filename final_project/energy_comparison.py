import numpy as np
import matplotlib.pyplot as plt

lp = np.load("results/leapfrog_periodic/energy.npy")
ln = np.load("results/leapfrog_nonperiodic/energy.npy")
rp = np.load("results/rk4_periodic/energy.npy")
rn = np.load("results/rk4_nonperiodic/energy.npy")

# Tracking energy conservation before particles fly out
lns = ln[:850]

# plt.plot(lp)
print("Energy diff. leapfrog per.\t", np.abs((np.max(lp) - np.min(lp)) / lp[0]) * 100, "%")
print("Energy diff. leapfrog nonper.\t", np.abs((np.max(lns) - np.min(lns)) / lns[0]) * 100, "%")
print("Energy diff. rk4 per.\t\t", np.abs((np.max(rp) - np.min(rp)) / rp[0]) * 100, "%")
print("Energy diff. rk4 nonper.\t", np.abs((np.max(rn) - np.min(rn)) / rn[0]) * 100, "%")

