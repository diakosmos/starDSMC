"""
This is a straight traslation of the dsmc programs from Alejandro Garcia's book.

He has two programs, "dsmceq" and "dsmcne", which have a lot of overlap. I'm beginning
with dsmcne.

Subtle tweaks:

The MATLAB version of his program 'dscmne' was actually several programs together:
    dsmcne.m ->
        sorter.m
        colider.m
        mover.m
        sampler.m

Here, I have bundled all of these together as Julia functions within a module,
the module being named "dsmc", intended to contain both "dsmcne" and "dsmceq".
"""

module dsmc

using Random

function dsmcne()
    # Initialize constants (particle mass, diameter, etc.)
    boltz   = 1.3806e-23
    mass    = 6.63e-26
    diam    = 3.66e-10
    T       = 273.0
    density = 2.685e25
    L       = 1.0e-6
    Volume  = L^3
    print("Enter number of simulation particles: ")
    npart = parse(Int64,readline())
    eff_num = density * Volume / npart
    print("Each simulation particle represents $eff_num atoms.\n")
    mfp = Volume / (sqrt(2.0) * π * diam^2 * npart * eff_num)
    swmfp = L/mfp # not defined in original code
    print("System width is $swmfp mean free paths.\n")
    mpv = sqrt(2*boltz*T/mass)
    print("Enter wall velocity as Mach number: ")
    vwall_m = parse(Float64,readline())
    vwall = vwall_m * sqrt(5.0/3.0 * boltz*T/mass)
    mvwall = -vwall # not defined in original code
    print("Wall velocities are $mvwall and $vwall m/s.\n")

    # Assign random positions and velocities to particles
    seed!(239482) # from Random
    x = L * rand(npart)
    
end

end # module
