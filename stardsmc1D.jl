"""
This is my attempt to take my straight translation of Garcia's DSMC code, and
re-purose it for planetary disks (rings).
"""
#=
const output = IOBuffer()
using REPL
const out_terminal = REPL.Terminals.TerminalBuffer(output)
const basic_repl = REPL.BasicREPL(out_terminal)
const basic_display = REPL.REPLDisplay(basic_repl)
Base.pushdisplay(basic_display)
=#

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

function sorter!(sD::sortDataS, x, xmin, xmax )
    show_histogram = false
    # Find the cell address for each particle (this could probably be made cleaner in Julia)
    #@assert count(q -> q < xmin, x)==0
    #@assert count(q -> q > xmax, x)==0
    npart = sD.npart
    ncell = sD.ncell
    ell = xmax - xmin
    xx = (x .- xmin) / ell
    jx = floor.(xx*ncell) .+ 1
    jx = min.(jx, ncell*ones(npart)) # <ugly>
    jx = max.(jx, ones(npart)) # also ugly
    jx = Int64.(jx)

    # Count the number of particles in each cell
    sD.cell_n = zeros(Int64,ncell)
    for ipart in 1:npart
        sD.cell_n[ jx[ipart] ] += 1
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
    if show_histogram
        #print(sD.cell_n )
        display( bar(sD.cell_n, #normalize=true,
            title = "cell distribution.",
            xlabel = "cell", ylabel= "Number in cell", legend=false)) #vbin)
    end
end#sorter!


# Declare structure for statistical sampling
mutable struct sampDataS
    ncell::Int64
    nsamp::Int64
    ave_n::Array{  Int64,1}
    ave_u::Array{Float64,2}
    ave_T::Array{Float64,1}
    ave_visc::Array{Float64,1}
end
sampDataS(ncell,nsamp) = sampDataS(
    ncell,
    nsamp,
    zeros(Int64,ncell),
    zeros(      ncell,3),
    zeros(      ncell),
    zeros(      ncell),
)

function reset!(sD::sampDataS)
    sD.nsamp = 0
    sD.ave_n *= 0
    sD.ave_u *= 0.0
    sD.ave_T *= 0.0
    sD.ave_visc *= 0.0
end

"""
sampler:
    Used by dsmcne() to sample the number density, fluid velocity, and
    temperature in the cells.

    Mutable (inout):
        sampD               sampling data

    Input (in):
        x                   particle positions
        v                   particle velocities
        npart               number of particles
        L                   system size

    Outputs:
        -none-

"""
function sampler!(sampD::sampDataS, x, v, npart, xmin, xmax )
    # Compute cell location for each particle
    ncell = sampD.ncell
    ell = xmax - xmin
    xx = (x .- xmin) / ell
    jx = floor.(xx*ncell) .+ 1
    jx = min.(jx, ncell*ones(npart)) # <ugly>
    jx = max.(jx, ones(npart)) # also ugly
    jx = Int64.(jx)

    # There must be a more succint way of writing this in Julia...
    # The idea is to sum the squares along the :
    normsq = x -> sum(x.^2,dims=2)[:,1]

    # Initialize running sums of number, velocity and v^2
    sum_n  = zeros(Int64, ncell)
    sum_v  = zeros(       ncell, 3)
    sum_v2 = zeros(       ncell)

    # For each particle, accumulate running sums for its cell
    for ipart in 1:npart
        jcell = jx[ipart] #    # particle ipart is in cell jcell
        sum_n[jcell]    += 1
        sum_v[jcell,:]  += v[ipart,:]
        sum_v2[jcell]   += v[ipart,:]' * v[ipart,:]  # (or norm?)
    end

    # Use current sums to update sample number, velocity and temperature
    for i in 1:3
        sum_v[:,i] ./= sum_n[:]
    end
    sum_v2 ./= sum_n
    sampD.ave_n += sum_n
    sampD.ave_u += sum_v
    sampD.ave_T += sum_v2 - normsq(sum_v)
    sampD.nsamp += 1

    return nothing
end#sampler!


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
function colider!!!(v,crmax,selxtra,  tau,coeff,sD,ecor)
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
                # @assert k != kk # Should not ever be a problem...
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
                    vrel[1] = ecor * cr*cos_th                 # Compute post-collision
                    vrel[2] = ecor * cr*sin_th * cos(phi)      #    relative velocity
                    vrel[3] = ecor * cr*sin_th * sin(phi)
                    v[ip1,:] = vcm + 0.5 * vrel         # Update post-collision
                    v[ip2,:] = vcm - 0.5 * vrel         #    velocities
                end
            end # Loop over pairs
        crmax[jcell] = crm    # Update max relative speed
        end
    end # Loop over cells
    return col
end#colider!!!

function getTeff(x,v,Omega,npart)
    shear = -1.5 * Omega
    vback = shear*x * [0.0 1.0 0.0]
    vprime = v - vback
    E = 0.0
    for i in 1:npart
        E += vprime[i,1]^2 + vprime[i,2]^2 + vprime[i,3]^2
    end
    return E
end

function slow(x,v,Omega,f)
    shear = -1.5 * Omega
    vback = shear*x * [0.0 1.0 0.0]
    vprime = v - vback
    vprime2 = f * vprime
    w = vprime2 + vback
    return w
    E = 0.0
end

function dsmcne()
    # Initialize constants (particle mass, diameter, etc.)
    boltz   = 1.3806e-23       # Boltzmann's constant (J/K)
    mass    = 6.63e-26         # Mass of argon atom (kg)
    diam    = 5.0e-10 # 3.66e-10         # Effective diamter of argon atom (m)
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
    print("Mean free path: $mfp\n")
        ncell = 256                     # Number of cells
    c_vs_mfp = (L / ncell)/mfp
    print("cell size / mfp (<1?): $c_vs_mfp\n")
    mpv = sqrt(2*boltz*T/mass)
    collfreq = mpv/mfp
    print("Collision frequency: $collfreq\n")
    print("Enter coefficient of restitution (e), 0<e<=1: ")
    ecor = parse(Float64,readline())
    @assert 0.0<ecor<=1.0
    print("Enter Omega as a fraction of mean collision frequency: ")
    Omega = parse(Float64,readline()) * collfreq
    print("Omgea = $Omega\n")
    Depi = mpv / 2 / pi / Omega
    print("Typical epicyclic diameter: $Depi\n")
    Depi_v_mfp = Depi / mfp
    print("Epicyclic diameter in mean free paths: $Depi_v_mfp\n")
    Depi_v_c = Depi / (L / npart)
    print("Epicyclic diameter / cell width: $Depi_v_c\n")
    # Assign random positions and velocities to particles
    Random.seed!(239482) # from Random - not exported by Random, apparently?
    x = L * rand(npart) .- L/2
    # Assign thermal velocities using Gaussian random numbers
    v = sqrt(boltz*T/mass) * randn(npart,3)
    # Add velocity gradient to the y-component
    v[:,2] += -1.5*Omega*x

    # Initialize variables used for evaluating collisions
    ncell = 256
    # Number of cells
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    tau = 0.123456 * 0.2 * (L/ncell) / mpv    # Set timestep tau
    print("Timestep (tau): $tau\n")
    cpertau = collfreq * tau
    print("Collisions per timestep: $cpertau\n")
    vrmax = 3*mpv*ones(ncell)#,1)  # Estimated max rl. speed in a cell
    selxtra = zeros(ncell)#,1)     # Used by collision routing "colider"
    coeff = 0.5 * eff_num * π * diam^2 * tau / (Volume/ncell)

    # Declare structure for lists used in sorting
    sortData = sortDataS(ncell,npart)
    sampData = sampDataS(ncell,0)
    tsamp = 0.0
    dvtot = [0.0, 0.0] #zeros(2)
    dverr = [0.0, 0.0] #zeros(2)

    # Loop for the desired number of time steps
    colSum = 0
    strikeSum = [0, 0]
    print("Enter total number of timesteps: ")
    nstep = parse(Int64,readline())

    # Initial sort the particles into cells
    sorter!(sortData, x, -L/2, L/2)
    E0 = getTeff(x,v,Omega,npart)
    for istep in 1:nstep
        # Periodically display the current progress, then reset samples:
        # Move all the particles
        (x,v,strikes, delv) = newmover!(x,v,npart,L,mpv,Omega,tau)
        #(strikes, delv) = mover!!(x,v,npart,L,mpv,vwall,tau)
        strikeSum += strikes

        # Sort the particles into cells
        sorter!(sortData, x, -L/2, L/2)

        # Evaluate collisions among the particles
        col = colider!!!(v, vrmax, selxtra,      tau, coeff, sortData, ecor)
        colSum += col

        # After initial transient, accumulate statistical samples
         #if istep > nstep/10
            sampler!(sampData, x, v, npart, -L/2, L/2)
            #dvtot += delv
            #dverr += delv.^2
            tsamp += tau

        E = getTeff(x,v,Omega,npart)
        f = sqrt(E0/E)
        v=slow(x,v,Omega,f)
        E = getTeff(x,v,Omega,npart)
        eoe0 = E / E0
        #print("E/E0: $eoe0\n")
        #end
        if( istep%1000 == 0)
            nsamp = sampData.nsamp
            ave_n = (eff_num/(Volume/ncell)) * sampData.ave_n / nsamp
            xcell = (collect(1.0:ncell).-0.5)/ncell * L .- L/2
            display(plot(xcell,ave_n,    # title = "This is the title.",
                xlabel = "position", ylabel= "Number density", legend=false))
            reset!(sampData)
        end


    end

    # Normalize the accumulated statistics
    nsamp = sampData.nsamp
    ave_n = (eff_num/(Volume/ncell)) * sampData.ave_n / nsamp
    ave_u = sampData.ave_u / nsamp
    ave_T = mass / (3*boltz) * (sampData.ave_T/nsamp)
    dverr = dverr /(nsamp-1) - (dvtot/nsamp).^2
    dverr = sqrt.(dverr*nsamp)

    # Plot average density, velocity and temperature
    xcell = (collect(1.0:ncell).-0.5)/ncell * L .- L/2
    display(plot(xcell,ave_n,    # title = "This is the title.",
        xlabel = "position", ylabel= "Number density", legend=false))
    display(plot(xcell,ave_u,    # title = "This is the title.",
        xlabel = "position", ylabel= "Velocities",
        label=["x-component" "y-component" "z-component"]))
    display(plot(xcell,ave_T,    # title = "This is the title.",
        xlabel = "position", ylabel= "Temperature", legend=false))
    return 0
end#dscmne

function newmover!(        x    ::Array{Float64,1},
        v    ::Array{Float64,2},
        npart::Int64,
        L    ::Float64,
        mpv  ::Float64,
        Omega::Float64,
        tau  ::Float64, )

    #Omega = 1.0e10



            #=
    v[:,1] +=  - 0.5 * tau * Omega^2 * x;

    Py = v[:,2] + 2*Omega*x
    #
    v[:,1] += tau * Omega * Py
    v[:,2]  = Py - Omega * x - Omega*(x + tau*v[:,1])
    #
    x += tau * v[:,1]
    #y += tau * v[:,2]
    #
    v[:,1] += tau * Omega * Py
    #
    v[:,1] -= 0.5 * tau * (Omega^2 * x)
    #xdot_[i+1] -= 0.5 * tau * (Omega^2 * xtest);
    v[:,2]   = Py - 2*Omega*x
    =#
    #
    Py = 0.0 * x # Just to allocate
    Base.Threads.@threads for i in 1:npart
    v[i,1] +=  - 0.5 * tau * Omega^2 * x[i];

    Py[i] = v[i,2] + 2*Omega*x[i]
    #
    v[i,1] += tau * Omega * Py[i]
    v[i,2]  = Py[i] - Omega * x[i] - Omega*(x[i] + tau*v[i,1])
    #
    x[i] += tau * v[i,1]
    #y += tau * v[:,2]
    #
    v[i,1] += tau * Omega * Py[i]
    #
    v[i,1] -= 0.5 * tau * (Omega^2 * x[i])
    #xdot_[i+1] -= 0.5 * tau * (Omega^2 * xtest);
    v[i,2]   = Py[i] - 2*Omega*x[i]
        if (x[i] < -0.5*L)
            x[i] += L;
            #part.y -= mod( 1.5*Omega*Lx*t, Ly);
            v[i,2] -= 1.5 * Omega * L;
        elseif (x[i] ≥ 0.5*L)
            x[i] -= L;
            #part.y += mod( 1.5*Omega*Lx*t, Ly);
            v[i,2] += 1.5 * Omega * L;
        end
        #part.y = mod( part.y,Ly);
    end
    return (x, v, [0.0,0.0],0.0)
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

end # module

dsmc.dsmcne()
