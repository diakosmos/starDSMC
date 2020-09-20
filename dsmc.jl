"""
This is a straight traslation of the dsmc programs from Alejandro Garcia's book.

He has two programs, "dsmceq" and "dsmcne", which have a lot of overlap.

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
using LinearAlgebra # for norm() of a vector
using Plots

"""
Declare structure for lists used in sorting:
    See Fig 11.7 in Garcia, which shows how Xref, cell_n and index should behave.
"""
mutable struct sortDataS
    ncell ::Int64
    npart ::Int64
    cell_n::Array{Int64,1}
    index ::Array{Int64,1}
    Xref  ::Array{Int64,1}
end
sortDataS(ncell,npart) = sortDataS(
    ncell,
    npart,
    zeros(ncell),
    zeros(Int64,ncell),
    zeros(Int64,npart),
)

function sorter!(sD::sortDataS, x, L )
    # Find the cell address for each particle (this could probably be made cleaner in Julia)
    npart = sD.npart
    ncell = sD.ncell
    jx = floor.(x*ncell/L) .+ 1
    jx = min.(jx, ncell*ones(npart)) # <ugly>
    jx = Int64.(jx)

    # Count the number of particles in each cell
    sD.cell_n = zeros(Int64,ncell)
    for ipart in 1:npart
        sD.cell_n[ jx[ipart]] += 1
    end

    # Build index list as a cumulative sum of the
    # number of particles in each cell.
    m = 1
    for jcell in 1:ncell
        sD.index[jcell] = m
        m += sD.cell_n[jcell]
    end

    # Build cross-reference list
    temp = zeros(Int64,ncell)
    for ipart in 1:npart
        jcell = jx[ipart]
        k = sD.index[jcell] + temp[jcell]
        sD.Xref[k] = ipart
        temp[jcell] += 1
    end
end


# Declare structure for statistical sampling
mutable struct sampDataS
    ncell::Int64
    nsamp::Int64
    ave_n::Array{Float64,1}
    ave_u::Array{Float64,2}
    ave_T::Array{Float64,1}
end
sampDataS(ncell,nsamp) = sampDataS(
    ncell,
    nsamp,
    zeros(ncell),
    zeros(ncell,3),
    zeros(ncell),
)


"""
colider():
    Called by dsmceq() and dsmcne() to evaluate collisions using the DSMC algorithm.

    NOTE the change in order from Garcia, so that mutated arguments are put first.

Updated (inout) Inputs:
    v           velocities of particles
    crmax       estmated maximum relative speed in a cell
    selxtra     extra selections carried over from last timestep

Non-mutated (in) inputs:
    tau         time step
    coeff       coefficient in computing number of selected pairs
    sD          structure containing sorting lists

Output:
    col         Total number of collisions processed
"""
function colider!!!(v,vrmax,selextra,  tau,coeff,sD)
    ncell = sD.ncell
    col = 0             # Count number of collisions

    # Loop over cells, processing collisions in each cell
    for jcell in 1:ncell
        # Skip cells with only one particle
        number = sD.cell_n[jcell]
        if number > 1
            # Determine number of candidate collision pairs
            # to be selected in this cell
            select = coeff * number * (number-1) *crmax[jcell] + selxtra[jcell]
            nsel = Int64(floor(select))
            selxtra[jcell] = select - nsel  # Carry over any left-over fraction
            crm = crmax[jcell]

            # Loop over total number of candidate collision pairs
            for isel in 1:nsel
                # Pick two particles at random out of this cell
                # NB: k, kk ∈ [0, 1, 2, ..., number-1], k != kk.
                k  = Int64(floor(rand() * number))
                kk = Int64(ceil(k+rand()*(number-1))) % number
                assert k != kk # Should not ever be a problem...
                ip1 = sD.Xref[k + sD.index[jcell]]  # First particle
                ip2 = sD.Xref[kk+ sD.index[jcell]]  # Second particle

                # Calculate pair's relative speed
                cr = norm( v[ip1,:] - v[ip2,:] )  # Relative speed
                if cr > crm   # If relative speed is greater than crm,
                    crm = cr  # then reset crm to larger value
                end

                # Accept or reject candidate pair according to relative speed
                if cr/crmax[jcell] > rand()
                    # If pair selected, then select post-collision velocities
                    col += 1                            # Collision counter
                    vcm = 0.5*(v[ip1,:] + v[ip2,:])     # Center-of-mass velocity
                    cos_th = 1 - 2*rand()               # Cosine and sine of
                    sin_th = sqrt(1.0 - cos_th^2)       #   collision angle theta
                    phi = 2π * rand()                   # Collsion angle phi
                    vrel = zeros(3)
                    vrel[1] = cr*cos_th                 # Compute post-collision
                    vrel[2] = cr*sin_th * cos(phi)      #    relative velocity
                    vrel[3] = cr*sin_th * sin(phi)
                    v[ip1,:] = vcm + 0.5 * vrel         # Update post-collision
                    v[ip2,:] = vcm - 0.5 * vrel         #    velocities
                end
            end # Loop over pairs
        crmax[jcell] = crm    # Update max relative speed
        end
    end # Loop over cells
    return col
end

"""
dsmceq():
    Dilute gas simulation using DSMC algorithm.
    This version illustrates the approach to equilibrium.
"""
function dsmceq()
    # Initialize constants (particle mass, diameter, etc.)
    boltz   = 1.3806e-23       # Boltzmann's constant (J/K)
    mass    = 6.63e-26         # Mass of argon atom (kg)
    diam    = 3.66e-10         # Effective diamter of argon atom (m)
    T       = 273.0            # Initial temperature (K)
    """ NOTE: in dsmcne(), for some reason, "density" is number density, whereas
    in dscmeq(), it's mass density. """
    density = 1.78             # Density of argon at STP (m^-3)
    L       = 1.0e-6           # System size is one micron
    print("Enter number of simulation particles: ")
    npart = parse(Int64,chomp(readline()))
    eff_num = density/mass*L^3/npart
    print("\nEach particle represents $eff_num atoms.\n")

    # Assign random positions and velocities to particles.
    Random.seed!(9827342423)      # initialize random number generator
    x = L*rand(npart)             # assign random positions
    v_init = sqrt(3*boltz*T/mass) # initial speed
    v = zeros(npart,3)            # only x-component is non-zero
    v[:,1] = v_init * (1 .- 2*floor.(2*rand(npart)))  # Makes some +, others -.

    # Plot the initial speed distribution
    vmag = sqrt.(v[:,1].^2 + v[:,2].^2 + v[:,3].^2)
    vbin = 50:100:1050      # Bins for histogram
    display(histogram(vmag, bins=vbin, #normalize=true,
        title = "Initial speed distribution.",
        xlabel = "Speed [m/s]", ylabel= "Number", legend=false)) #vbin)

    # Initialize variables used for evaluating collisions.
    ncell = 15                    # Number of cells
    tau   = 0.2*(L/ncell)/v_init  # Set timestep tau
    vrmax = 3*v_init*ones(ncell)  # Estimated max relative speed
    selxtra = zeros(ncell)        # Used by routine "colider"
    coeff = 0.5*eff_num*pi*diam^2*tau/(L^3/ncell)
    coltot = 0

    # [ next, Garcia declares structure for lists used in sorting, but we have
    #  moved this to the preamble above. ]

    # Loop for the desired number of time steps.
    print("Enter total number of time steps: ")
    nstep = parse(Int64,chomp(readline()))
    for istep in 1:nstep

        # Move all the particles ballistically
        x += v[:,1]*tau   # Update x position of particles - recall x is a 1D array
        x = rem.(x+L,L)   # Periodic boundary conditions (why does Garcia add L first?!?!)

        # Sort the particles into cells
        sortData = sorter(x,L,sortData)

        # Evaluate collisions among the particles
        (v, vrmax, selextra, col) = colider(v, vrmax, tau, selextra, coeff, sortData)
        coltot += col

        # Periodically display the current progress
        if( istep%10 < 1)
            vmag = sqrt.(v[:,1].^2 + v[:,2].^2 + v[:,3].^2)
            display(histogram(vmag, bins=vbin, #normalize=true,
                title = "Done $istep of $nstep steps; $coltot collisions.",
                xlabel = "Speed [m/s]", ylabel= "Number", legend=false))
        end
    end

    # Plot the histogram of the final speed distribution
    vmag = sqrt.(v[:,1].^2 + v[:,2].^2 + v[:,3].^2)
    time = nstep*tau
    display(histogram(vmag, bins=vbin, #normalize=true,
        title = "Final distrib., Time = $time sec.",
        xlabel = "Speed [m/s]", ylabel= "Number", legend=false))

    return nothing #vmag
end

function dsmcne()
    # Initialize constants (particle mass, diameter, etc.)
    boltz   = 1.3806e-23       # Boltzmann's constant (J/K)
    mass    = 6.63e-26         # Mass of argon atom (kg)
    diam    = 3.66e-10         # Effective diamter of argon atom (m)
    T       = 273.0            # Initial temperature (K)
    """ NOTE: in dsmcne(), for some reason, "density" is number density, whereas
    in dscmeq(), it's mass density. """
    density = 2.685e25         # Number density of argon at STP (m^-3)
    L       = 1.0e-6           # System size is one micron
    Volume  = L^3              # Volume of the system (m^3)
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
    Random.seed!(239482) # from Random - not exported by Random, apparently?
    x = L * rand(npart)
    # Assign thermal velocities using Gaussian random numbers
    v = sqrt(boltz*T/mass) * rand(npart,3)
    # Add velocity gradient to the y-component
    v[:,2] += 2*vwall*(x/L) .- vwall

    # Initialize variables used for evaluating collisions
    ncell = 20                     # Number of cells
    tau = 0.2 * (L/ncell) / mpv    # Set timestep tau
    vrmax = 3*mpv*ones(ncell)#,1)  # Estimated max rl. speed in a cell
    selxtra = zeros(ncell)#,1)     # Used by collision routing "colider"
    coeff = 0.5 * eff_num * π * diam^2 * tau / (Volume/ncell)

    # Declare structure for lists used in sorting
    sortData = sortDataS(ncell,npart)
    sampData = sampDataS(ncell,0)
    tsamp = 0.0
    dvtot = zeros(2)
    dverr = zeros(2)

    # Loop for the desired number of time steps
    colSum = 0
    strikeSum = [0, 0]
    print("Enter total number of timesteps: ")
    nstep = parse(Int64,readline())
    for istep in 1:nstep
        (strikes, delv) = mover!(x,v,npart,L,mpv,vwall,tau)
        strikeSum += strikes
    end
    return 0
end

"""
mover:
    - Function to move particles by free flight. Also handles collisions with walls.

    inputs:
    x       positions of the particles
    v       velocities of the particles
    npart   number of particles in the system
    L       system length
    mpv     most probable velocity (off the wall)
    vwall   wall velocities
    tau     time step

    outputs:
    [ x, v    updated positions and velocities ] - actually not output; mutated.
    strikes number of particles striking each wall
    delv    change of y-velocity at each wall
"""
function mover!(
        x    ::Array{Float64,1},
        v    ::Array{Float64,2},
        npart::Int64,
        L    ::Float64,
        mpv  ::Float64,
        vwall::Float64,
        tau  ::Float64,
    )  # return: (x, v, strikes, delv)
    x_old = x
    x += v[:,1] * tau
    strikes = [0,0]
    delv = [0.0,0.0]
    xwall = [0.0 L]
    vw = [-vwall, vwall]
    direction = [1,-1]
    stdev = mpv / sqrt(2.0)
    for i in 1:npart
        # Test if particle strikes either wall
        if x[i] <= 0
            flag = 1 # particle strikes left wall
        elseif x[i] >= L
            flag = 2 # particle strikes right wall
        else
            flag = 0 # particle strikes neither wall
        end

        # If particle strikes a wall, reset its position
        # and velocity. Record velocity change.
        if flag > 0
            strikes[flag] += 1
            vyInitial = v[i,2]
            # Reset velocity components as biased Maxwellian
            # Exponential dist. in x; Gaussian in y an z
            v[i,1] = direction[flag] * sqrt(-log(1-rand())) * mpv
            v[i,2] = stdev*randn() + vw[flag] # Add wall velocity
            v[i,3] = stdev*randn()
            # Time of flight after leaving wall
            x[i] = xwall[flag] + v[i,1]*dtr
            # Record velocity change for force measurement
            delv[flag] += v[i,2] - vyInitial
        end
    end
    #return (x,v,strikes,delv)
    return (strikes,delv) # No need to return x, v, since mutable & passed by sharing.
end

end # module
