# starDSMC
DSMC implemented for planetary rings etc

- old_first_try.jl: my first lame attempt at coding this up - but, don't delete. It took a while to get the dynamics right, and the modulo arithmetic right (it's a bit more tricky than you think); this is basically cut-and-pasted from a jupyter julia notebook where I spent a long time experimenting and satisfying myself that it worked properly.

- dsmc_Garcia_0.jl: basically a straight translation of Garcia's dsmceq and dsmcne codes in his book.

- stardsmc1D.jl: a 1D DSMC code based tightly on Garcia's, but with periodic BC and coriolis & shear. Shows instability when epicyclic frequency is comparable to collision frequency. Instability may manifest as traveling or stationary waves, with some structure to the waves (e.g. twin peaks). Try 123456 particles, omega = pi * collision_freq, and 123456 timesteps.

