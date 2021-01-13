module rings

using SpecialFunctions
using Random
using Test
using LinearAlgebra # for norm
using Plots

#=macro doit(expn)
    eval(expn)
end=#

"""
Units: cgs

Note to self re parallelization:

see https://discourse.julialang.org/t/struct-of-arrays-soa-vs-array-of-structs-aos/30015

Quoting:
@inbounds , @simd , @simd ivdep
check if it has been vectorized with @code_llvm.

Using BenchmarkTools
@btime

Some fiducial quantities for B-ring:
    https://nssdc.gsfc.nasa.gov/planetary/factsheet/satringfact.html

    Thickness about 10m?
    Optical depth: 0.2-5 (say, 3)
    Particle size: 1m diam? So mass is about 5.236e5 g.
    Surface density abou 100g/cm^2
    Particle interior density 1g/cm^3
    number density about 2e-7 / cm^3
    orbital radius 106,000 km (92,000   –  117,580) = 1.06e10 cm
    Saturn mass: 5.6834e29 g
    G = 6.6743e-8 in cgs, so
    Omega about 1,7846e-4 / s, i.e. a period of about 9.8hr.
    For scale ht H = 10m, then, typical peculiar velocity is Omega*H =
    0.1785 cm/s (i.e. about 2mm/s)

TESTS:
    + cell size < mean-free-path (must resolve a mfp)
    + time step << collision timescale
    + time step << orbital timescale
    + number of simulated particles per cell > 20
    + background (shear) velocity difference across cell < peculiar velocity v'
    + mfp >> (particle diameter)
    + no. of particles in (mfp)^3 (>>1? Or, irrelevant?)
"""

fiducials = (
    Ω = 1.7846e-4, # angular frequency in middle of B-ring
    τ = 3.0,   # optical depth
    D = 1.0e2, # 1 m
    Σ = 1.0e2, # g/cm^2
    ρ = 1.0,   # g/cm^3
    m = 5.236e5, # =(π/6)*D^3, ≈ 523kg.
    n = 2.0e-7,# /cm^3
    N = 3.82e-4, # = Σ / m, /cm^2
    σ = 7854.0, # (π/4)*D^2
    H = 1.0e3, # cm
    v′= 0.1785, # cm/s (peculiar velocity)
    R₀= 1.06e10, # orbital radius
    #
    xmin = 1.0e2, #floor for 1/x^4 to avoid divergence
    ablah = 1.0e-5,
    xH = 1.0e4,
)

Random.seed!(21237428012)

myint = Int64
"""
################################################################################
struct mesh: an immutable struct for recording the size and location of
    the Cartesian mesh used to sort particles.

    Also, stores continuum quantities.
"""
struct mesh
    nx::myint # number of CELLS (not nodes) in x-dir
    ny::myint # etc
    nz::myint
    x_::Tuple{Vararg{Float64}} # x-location of cell boundaries (i.e. nodes)
    y_::Tuple{Vararg{Float64}} # Set as tuples, to enforce immutability
    z_::Tuple{Vararg{Float64}}
    Hz::Float64 # scale ht used to set z_.
    #
    Vc_::Array{Float64,3} # cell volume
    mfp_::Array{Float64,3} # mean-free-path (estimate)
    Lmfp_::Array{Float64,3}
    collfreq_::Array{Float64,3} # collision frequency (estimate)
    #
    crmax_::Array{Float64,3} # estimate of maximum relative speed in a cell
    selxtra_::Array{Float64,3} # extra selections carried over from last timestep
    coeff_::Array{Float64,3} # coeff related to collision freq in a cell
end
"""
 Nominal outer constructor. Note Hz is vertical scale height.
"""
function mesh(nx,ny,nz,Lx::Real,Ly::Real,Hz::Real)
    @assert nz>=3
    if  isfinite(Ly)
        y = (collect(range(-Ly/2,Ly/2;length=ny+1))...,) # <-- this syntax constructs tuple from array
    else
        @assert ny == 1
        y = (-Inf,Inf)
    end
    print("Boundaries in vertical direction: \n")
    print((Hz * SpecialFunctions.erfinv.(collect(range(-1.0,1.0;length=nz+1)))...,))
    print("\n")
    mesh( nx, ny, nz,
        (collect(range(-Lx/2,Lx/2;length=nx+1))...,),
        y,
        (Hz * SpecialFunctions.erfinv.(collect(range(-1.0,1.0;length=nz+1)))...,),
        Hz,
        zeros(nx,ny,nz),
        zeros(nx,ny,nz),
        zeros(nx,ny,nz),
        zeros(nx,ny,nz), # collfreq_
        zeros(nx,ny,nz), # crmax_
        zeros(nx,ny,nz), # selxtra_
        zeros(nx,ny,nz), # coeff_
    )
end
""" Convenience 2D (x,z) constructor: """
mesh(nx::Int,nz::Int,Lx::Real,Hz::Real) = mesh(nx,1,nz,Lx,2*π*fiducials.R₀,Hz)
# Note: setting the y-limits infinite causes problems. Why? B.c then y-positions
# aren't initialized to something reasonable; instead, they are NaNs. This messes
# up index counting (binning) based on y-values.

"""
################################################################################
struct particles:

    Note: Struct-of-Arrays faster than Array-of-Structs

    Note: in action-angle variables for harmonic oscillator (Hamilton-Jacobi), canonical action is
    equal to total energy E = (1/2)mv^2 + (1/2)kz^2. Then
    E = (1/2) m^2 Omega^2 z_max^2  and z_max = sqrt(2*E) / (m * Omega).

"""

struct particles # Hmmm - should hand it a mesh, not set its ranges here.
    N::myint # number of particles
    Neff::Float64 # number of real particles per simulated particle
    Omega::Float64 # epicyclic freq
    shear::Float64 # nominally -(3/2)Ω
    #m_::Tuple{Vararg{Float64}} # mass
    m::Float64
    #D_::Tuple{Vararg{Float64}} # Diameter
    D::Float64
    #Lx_  ::Array{Float64,1} # x-component of rotational angular momentum
    #Ly_  ::Array{Float64,1}
    #Lx_  ::Array{Float64,1}
    # Position arrays:
    x_::Array{Float64,1}
    y_::Array{Float64,1}
    z_::Array{Float64,1}
    # Velocity arrays:
    vx_::Array{Float64,1}
    vy_::Array{Float64,1}
    vz_::Array{Float64,1}
    # Momentum arrays (what are the conserved quantities?):
    Py_::Array{Float64,1} # canonical conserved quantity equivalent to angular momentum
    Ez_::Array{Float64,1} # conserved action corresponding to vertical oscillations: (1/2)m^2 Omega^2 (zmax)^2
    Thz_::Array{Float64,1}
end
function particles(N::Int,Neff::Float64, Omega::Float64,shear::Float64,m::Real,D::Real,
    xlim::NTuple{2,Real}, ylim::NTuple{2,Real}, Hz::Real,
    dvx::Real, dvy::Real, dvz::Real )
    # Note: zlim is not really a "limit" here so much as a scale height.
    m_ = (m*ones(N)...,)
    D_ = (D*ones(N)...,)
    x_ = xlim[1] .+ (xlim[2]-xlim[1])*Random.rand(N)
    y_ = ylim[1] .+ (ylim[2]-ylim[1])*Random.rand(N)
    z_ = Hz*Random.randn(N)
    vx_= dvx*Random.randn(N)
    vy_= dvy*Random.randn(N) + shear*x_
    vz_= dvz*Random.randn(N)
    Py_= zeros(N)
    Ez_= zeros(N)
    Thz_ = zeros(N)
    #particles(N,m_,D_,x_,y_,z_,vx_,vy_,vz_,Py_,Ez_)
    particles(N,Neff,Omega,shear,m,D,x_,y_,z_,vx_,vy_,vz_,Py_,Ez_,Thz_)
end
function particles(N::Int,Neff::Float64,Omega::Float64,shear::Float64,m::Real,D::Real,
    dv::Real,M::mesh)
    xx = (M.x_[1],M.x_[end])
    yy = (M.y_[1],M.y_[end])
    Hz = M.Hz #NO!
    Hz = abs(dv/Omega)
    particles(N,Neff,Omega,shear,m,D,xx,yy,Hz,dv,dv,dv)
end


"""
################################################################################
struct sortData:
    This is done as set of linear (flat) arrays, even tho the underlying
    mesh is 3D.
"""
struct sortData
    M     ::myint # re M and N: (1) should rename to ncell and npart, or
    N     ::myint # (2) just get rid of. (It appears I don't really need them.)
    #
    # particle arrays:
    cx_   ::Array{myint,1} # cell index. putting this here avoids constantly re-allocating
    Xref_::Array{myint,1}
    #
    # Mesh arrays:
    ncell_::Array{myint,3} # ncell_[i] = no of particles in cell "i"
    index_::Array{myint,3} # just cumsum of ncell_, really
    #
end
#=function sortData(ncell::Int64,npart::Int64)
    sortData(
        ncell,
        npart,
        zeros(myint,npart),
        zeros(myint,ncell),
        zeros(myint,ncell),
        zeros(myint,npart)
    )
end =#
function sortData(p::particles, m::mesh)
    ncell = m.nx * m.ny * m.nz
    npart = p.N
    sortData(
        ncell,
        npart,
        zeros(myint,npart),
        zeros(myint,npart),
        zeros(myint,m.nx, m.ny, m.nz), #ncell),
        zeros(myint,m.nx, m.ny, m.nz), #ncell),
    )
end
""" convenience function so argument order doesn't matter: """
function sortData(m::mesh,p::particles)
    sortData(p,m)
end

function sorter!(sD::sortData, P::particles, m::mesh)
    cx_ = sD.cx_ # <-- make sure this is an alias, not a copy (right?)
    nx = m.nx; ny = m.ny; nz = m.nz; N = P.N; M = nx*ny*nz
    @assert sD.M == M
    @assert sD.N == N
    # I think there might be library utility functions available for these two...:
    function ix(i::Int,j::Int,k::Int)
        @assert 0 < i <= nx
        @assert 0 < j <= ny
        @assert 0 < k <= nz
        i + nx*(j-1) + nx*ny*(k-1)
    end
    function ijk(ℓ::Int)
        @assert 0 < ℓ <= M
        i  = (ℓ -1) % nx + 1
        jk = (ℓ -1) ÷ nx + 1
        j  = (jk-1) % ny + 1
        k  = (jk-1) ÷ ny + 1
        (i,j,k)
    end
    """ (1) build up jx, i.e., find cell for each particle """
    Base.Threads.@threads     for p in 1:N
        x = P.x_[p]
        y = P.y_[p]
        z = P.z_[p]
        i = sum(map(q->q<=x,m.x_))
        #@assert 0<i<=m.nx
        j = 1 #sum(map(q->q<=y,m.y_))
        #@assert 0<j<=m.ny
        k = sum(map(q->q<=z,m.z_))
        #@assert 0<k<=m.nz
        #ixx = ix(i,j,k)
        #print("i: $i,  j: $j,  k: $k,  ix: $ixx\n")
        cx_[p] = ix(i,j,k)
    end
    """ (2) count particles in each cell """
    sD.ncell_[:] .= 0 # zero it out before starting the count
    @simd for p in 1:N
        sD.ncell_[cx_[p]] += 1
    end
    nc=sD.ncell_
    #print("sD.ncell_: $nc\n")
    s = sum(nc)
    #print("sum: $s\n")
    """ (3) build index list (cumsum) """
    sD.index_[1] = 1
    sD.index_[2:end] = 1 .+ cumsum(sD.ncell_[:])[1:end-1]
    cs=sD.index_
    #print("sD.index_: $cs\n")
    """ (4) build cross-reference list """
    temp_ = zeros(myint,M)
    #Base.Threads.@threads
    @simd for p in 1:N
        c = cx_[p]
        k = sD.index_[c] + temp_[c]
        sD.Xref_[ k ] = p
        #X = sD.Xref_
        #print("p: $p,  c: $c,  k: $k\n" )
        #print("sD.Xref_: $X\n")
        temp_[c]    += 1
    end
end


"""
mesh+particles functions
"""

"""
##################### The following functions need to be consolidated. #########
"""

function getcellvolume!(M::mesh)
    function dif(x)
        diff(collect(x))
    end
    for ℓ in eachindex(M.Vc_)
        i  = (ℓ -1) % M.nx + 1
        jk = (ℓ -1) ÷ M.nx + 1
        j  = (jk-1) % M.ny + 1
        k  = (jk-1) ÷ M.ny + 1
        M.Vc_[ℓ] = dif(M.x_)[i] * dif(M.y_)[j] * dif(M.z_)[k]
    end
end

function setEz!(P::particles)
    Ω = P.Omega; m=P.m
    for i in eachindex(P.z_)
        P.Ez_[i] = 0.5 * m * P.vz_[i]^2 + 0.5 * m * Ω^2 * P.z_[i]^2
        vzmax = √(2*P.Ez_[i]/m)
        zmax = vzmax / Ω
        Z = P.z_[i] / zmax
        VZ = P.vz_[i] / vzmax
        P.Thz_[i] = atan(Z,VZ)
        # Also:
        P.Py_[i]  =  P.vy_[i] + 2Ω*P.x_[i]
    end
end

function getmfp!(P::particles,M::mesh,sD::sortData)
    sorter!(sD,P,M) # updates sD.ncell_
    getcellvolume!(M)
    D = P.D
    for i in eachindex(M.collfreq_)
        M.mfp_[i] = M.Vc_[i] /(√2.0 * π * D^2 * sD.ncell_[i] *P.Neff)
    end
    for c in CartesianIndices(M.collfreq_)
        i=c[1]; j=c[2]; k=c[3]
        Lx = M.x_[i+1]-M.x_[i]
        Ly = M.y_[j+1]-M.y_[j]
        Lz = M.z_[k+1]-M.z_[k]
        L = max(Lx,Ly,Lz)
        M.Lmfp_[c] = L / M.mfp_[c]
    end
    #@assert minimum(M.Lmfp_[:,:,2:end-1]) > 1.0
end

function testcellsize(P::particles,M::mesh,sD::sortData)
    getmfp!(P,M,sD)
    for i in eachindex(M.mfp_)
        csize = max(M.x_[i+1]-M.x_[i], M.z_[])
    end
end

function getcollfreq!(M::mesh,P::particles)
    # representative velocity / mean_free_path
    for i in eachindex(M.collfreq_)
        M.collfreq_[i] = fiducials.v′ / M.mfp_[i] #not really
    end
end

function getcoeff!(M::mesh,P::particles,sD::sortData, tau::Float64 )
    M.coeff_[:] = 0.5 * P.Neff * π * P.D^2 * tau ./ M.Vc_
end

"""
###############################################################################
"""

function move!(P::particles,M::mesh,tau::Float64)
    Ω = P.Omega; x_=P.x_; y_=P.y_; z_=P.z_; vx_ = P.vx_; vy_=P.vy_; vz_=P.vz_; Py_=P.Py_
    xL = M.x_[1]; xU = M.x_[end]; Lx = xU-xL; Ez_ = P.Ez_; Thz_ = P.Thz_
    m = P.m
    # xy motion:
    @simd for i in eachindex(P.x_) # can use any of P's 1D length-N arrays.
        vx_[i] += -0.5 * tau * Ω^2 * x_[i]
        Py_[i]  =  vy_[i] + 2Ω*x_[i] # <-- since this is conserved, shouldn't be re-setting it each iteration.
        vx_[i] +=  tau * Ω * Py_[i]
        vy_[i]  =  Py_[i] - Ω * x_[i] - Ω*(x_[i] + tau*vx_[i])
        if x_[i] > 0
            x = max(x_[i],xmin)
            a = ablah * (xH/x)^4
            vy_[i] += a * tau
        end
        if x_[i] < 0
            x = min(x_[i],-xmin)
            a = ablah * (xH/x)^4
            vy_[i] -= a * tau
        end
        """ """
        x_[i]  +=  tau * vx_[i]
        y_[i]  +=  tau * vy_[i]
        """ """
        vx_[i] +=  tau * Ω * Py_[i]
        vx_[i] -=  0.5 * tau * (Ω^2 * x_[i])
        vy_[i]  =  Py_[i] - 2Ω*x_[i]
        if x_[i]<xL || x_[i]>=xU # could remove this conditional, actually...
            (k,xx) = fldmod(x_[i]-xL,Lx)
            x_[i] = xL + xx
            vy_[i] += k * 1.5*Ω*Lx #* 2.0/1.5
            Py_[i]  =  vy_[i] + 2Ω*x_[i] # Change?
        end#if
        if x_[i] > 0
            x = max(x_[i],xmin)
            a = ablah * (xH/x)^4
            vy_[i] += a * tau
        end
        if x_[i] < 0
            x = min(x_[i],-xmin)
            a = ablah * (xH/x)^4
            vy_[i] -= a * tau
        end
    end#for
    # z motion (should combine)
    @simd for i in eachindex(P.z_)
        if false # The method below is exact, but quite slow b/c of trig functions.
            # Float mult is 4-6 clock cycles AND can be pipelined (effectively, one per clock cycle);
            # float divide is 20-30 and can't be pipelined, and *trig* is about 200, and can't be pipelined.
            #Ez_[i] = 0.5 * m * vz_[i]^2 + 0.5 * m * Ω^2 * z_[i]^2
            vzmax = √(2*Ez_[i]/m)
            zmax = vzmax / Ω
            #= Z = z_[i] / zmax
            VZ = vz_[i] / vzmax
            θ = atan(Z,VZ) =#
            Thz_[i] += tau * Ω
            Z = sin(Thz_[i])
            VZ = cos(Thz_[i])
            z_[i] = Z * zmax
            vz_[i] = VZ * vzmax
        else
            az = -Ω^2 * z_[i]
            vz_[i] += az * tau
            z_[i] += vz_[i] * tau
        end
    end
end#function

function collide!(P::particles,M::mesh,sD::sortData, Dt::Float64; ecor::Float64=1.0,
    usephenom::Bool=false)
    # Note: ecor = elastic coefficient of (restition?) anyway, btwn 0 & 1.
    nx=M.nx; ny=M.ny; nz=M.nz; ncell=nx*ny*nz; vx_=P.vx_; vy_=P.vy_; vz_=P.vz_
    coll=0
    """
    phrest: phenomenological coefficient of restitution
    XXX PROBLEM: *actually* the velocity here is the NORMAL velocity,
    which is not the same as the relative velocity, b/c collisions are
    not generally head-on!!! FIX.
    """
    function phcr(v::Float64;vc=0.029,p=0.19)
        if v<=vc
            return 1.0
        else
            return (v/vc)^(-p)
        end
    end
    function pinch(x::Float64)
        z = 2.0*x-1.0
        return acos(-z)/π
    end
    getcoeff!(M,P,sD,Dt)
    Threads.@threads for jcell in 1:ncell
        no = sD.ncell_[jcell] # get number ("no") in cell
        if no>1
            select = M.coeff_[jcell] * no * (no-1) * 0.2
                # M.crmax_[jcell] + M.selxtra[jcell]
            nsel = Int64(floor(select))
            crm = M.crmax_[jcell] # XXX you never re-calculate this!!!
            # Loop over total number of candidate collision pairs
            for isel in 1:nsel
                k1 = Int64(floor(rand()*no))
                k2 = Int64(ceil(k1+rand()*(no-1))) % no
                @assert k1 != k2
                ip1 = sD.Xref_[k1 + sD.index_[jcell]]  # First particle
                ip2 = sD.Xref_[k2 + sD.index_[jcell]]  # Second particle
                # Calculate pair's relative speed
                cr = norm( (vx_[ip1] - vx_[ip2],
                            vy_[ip1] - vy_[ip2],
                            vz_[ip1] - vz_[ip2] ))  # Relative speed
                #print("Relative speed: $cr\n")
                if cr > crm   # If relative speed is greater than crm,
                    crm = cr  # then reset crm to larger value
                end
                # Accept or reject candidate pair according to relative speed
                if cr/M.crmax_[jcell] > rand()
                    # If pair selected, then select post-collision velocities
                    coll += 1                            # Collision counter
                    vcm = 0.5*[vx_[ip1] + vx_[ip2],
                                vy_[ip1] + vy_[ip2],
                                vz_[ip1] + vz_[ip2]]     # Center-of-mass velocity
                    cos_th = 1 - 2*pinch(pinch(pinch(pinch(pinch(rand())))))               # Cosine and sine of
                    sin_th = sqrt(1.0 - cos_th^2)       #   collision angle theta
                    phi = 2π * rand()                   # Collsion angle phi
                    vrel = zeros(3)
                    vrel[3] = cr*cos_th                 # Compute post-collision
                    vrel[2] = cr*sin_th * cos(phi)      #    relative velocity
                    vrel[1] = cr*sin_th * sin(phi)
                    if usephenom
                        ecor = phcr(norm(vrel);vc=0.5)
                    end
                    (vx_[ip1], vy_[ip1], vz_[ip1]) = vcm + 0.5 * ecor * vrel         # Update post-collision
                    (vx_[ip2], vy_[ip2], vz_[ip2]) = vcm - 0.5 * ecor * vrel         #    velocities
                end
            end
        M.crmax_[jcell] = crm
        end
    end
    return coll
end

""" Will move this eventually...."""
function main()
    # number of cells in z-dir, in x-dir, and total number of particles:
    Nz = 13; Nx = 512; Npart = 123456
    #
    f = fiducials
    Ω = f.Ω; D = f.D * 0.65; m=f.m; v′=f.v′; H=f.H
    s = -1.5 * Ω # shear
    Δx = 100.0 * H # Range in x-dir in terms of vertical scale heights
    Δt = 0.1 / Ω

    Nreal = Δx * 2*π*f.R₀ * f.N
    Neff  = Nreal / Npart
    print("Each simulated particle represents $Neff real particles.\n")

    M = rings.mesh(Nx,Nz, Δx, 4*H)
    P=rings.particles( Npart , Neff, Ω , s , m , D , v′ , M )
    sD = Main.rings.sortData(P,M)

    #for i in 1:1
    rings.sorter!( sD, P, M )
    for i in 1:12345
    rings.getmfp!(P,M,sD)
    rings.getcollfreq!(M,P)
    rings.getcoeff!(M,P,sD,Δt)
    rings.move!(P,M,Δt)
    rings.sorter!(sD, P, M)
    ncoll = rings.collide!(P,M,sD,Δt;usephenom=true)#ecor=0.90)
    print("Timestep: $i  Collisions: $ncoll.       ")
    cfreq = (ncoll*1.0)/(Npart*1.0) * 1.0/Δt
    omegaovercfreq = Ω/cfreq
    print("Ω / (collision frequency) = $omegaovercfreq\n")
    if i%10 == 1
        xx = [i for i in M.x_]
        xm = 0.5 * ( xx[1:end-1]+xx[2:end])
        display(plot(xm,sD.ncell_[:,1,5:9]))
        display(plot(P.z_,P.vz_,seriestype=:scatter))
        display(plot(P.x_,P.vx_,seriestype=:scatter))
        display(plot(P.x_,P.vy_,seriestype=:scatter))
    end
    end

    #
    # Now, to set timestep, need to estimate max collision freq.
    (sD,P,M)
end
end
