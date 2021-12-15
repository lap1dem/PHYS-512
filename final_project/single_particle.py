from particles import *
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# ========================
np.random.seed(172)
# ========================


GRIDSIZE = 128
DT = 0.1
NPART = 1
savedir = 'results/leapfrog_single/'

if not os.path.exists(savedir):
    os.makedirs(savedir)

parts = Particles(npart=NPART, soft=1, gridsize=GRIDSIZE, periodic=False)
parts.x[:] = GRIDSIZE / 2
parts.leapfrog_shift(DT/2)

energy = np.array([])

plt.ion()
rho_data = parts.rho.copy()
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.set_title("One particle")
ppart = ax.imshow(parts.rho, cmap='inferno')
xl, xr = ax.get_xlim()
yl, yr = ax.get_ylim()
ann = ax.annotate(r"$(2T - U)$ / $N_{part}^2$ = " + f"{0:.3e}",
                ((xr - xl) / 2, 0.06 * yl), fontsize=10,
                c='w', ha='center', va='center')

for i in tqdm(range(150)):
    kin_1 = np.sum(parts.v ** 2 / 2)
    parts.take_step(dt=DT)
    pot = -0.5 * np.sum(parts.rho * parts.pot)
    kin_2 = np.sum(parts.v ** 2 / 2)
    kin = (kin_1 + kin_2) / 2
    energy = np.append(energy, kin + pot)

    ppart.set_data(parts.rho)
    ann.set_text(r"$T + U$ = " + f"{energy[i]:.3e}")
    plt.pause(0.001)
    fig.savefig(savedir + f"{i:04d}", dpi=300, bbox_inches='tight')

np.save(savedir + 'energy', energy)

def compile_video(fps, name, imgdir='', codec='libx264', crf=25):
    os.system(f"ffmpeg -r {fps} -f image2 -s 1080x1080 -i {imgdir}%04d.png -vcodec {codec} -crf {crf}  -pix_fmt yuv420p {name}.mp4")

compile_video(30, 'one_particle', imgdir=savedir, codec='libx264')