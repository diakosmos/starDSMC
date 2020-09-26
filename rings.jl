module rings

using SpecialFunctions
using Random
using Test
"""
Units: cgs

Note to self re parallelization:

see https://discourse.julialang.org/t/struct-of-arrays-soa-vs-array-of-structs-aos/30015

Quoting:
@inbounds , @simd , @simd ivdep
check if it has been vectorized with @code_llvm.

Using BenchmarkTools
@btime
"""

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
    collfreq_::Array{Float64,3} # collision frequency (estimate)
end
"""
 Nominal outer constructor. Note Hz is vertical scale height.
"""
function mesh(nx,ny,nz,Lx::Real,Ly::Real,Hz::Real)
    if  isfinite(Ly)
        y = (collect(range(-Ly/2,Ly/2;length=ny+1))...,) # <-- this syntax constructs tuple from array
    else
        @assert ny == 1
        y = (-Inf,Inf)
    end
    mesh( nx, ny, nz,
        (collect(range(-Lx/2,Lx/2;length=nx+1))...,),
        y,
        (Hz * SpecialFunctions.erfinv.(collect(range(-1.0,1.0;length=nz+1)))...,),
        Hz,
        zeros(nx,ny,nz),
        zeros(nx,ny,nz),
        zeros(nx,ny,nz),
    )
end
""" Convenience 2D (x,z) constructor: """
mesh(nx::Int,nz::Int,Lx::Real,Hz::Real) = mesh(nx,1,nz,Lx,Inf,Hz)

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
    m_::Tuple{Vararg{Float64}} # mass
    D_::Tuple{Vararg{Float64}} # Diameter
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
end
function particles(N::Int,m::Real,D::Real,
    xlim::NTuple{2,Real}, ylim::NTuple{2,Real}, Hz::Real,
    dvx::Real, dvy::Real, dvz::Real, shear::Real )
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
    particles(N,m_,D_,x_,y_,z_,vx_,vy_,vz_,Py_,Ez_)
end
function particles(N::Int,m::Real,D::Real,dv::Real,shear::Real,M::mesh)
    xx = (M.x_[1],M.x_[end])
    yy = (M.y_[1],M.y_[end])
    Hz = M.Hz
    particles(N,m,D,xx,yy,Hz,dv,dv,dv,shear)
end

"""
mesh+particles functions
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

function getcollfreq!(M::mesh,P::particles)
    for i in eachindex(M.collfreq_)
        M.collfreq_[i] = M.Vc_[i] + 3.5 #not really
    end
end

"""
################################################################################
struct sortData:
    This is done as set of linear (flat) arrays, even tho the underlying
    mesh is 3D.
"""
struct sortData
    M     ::myint
    N     ::myint
    cx_   ::Array{myint,1} # cell index. putting this here avoids constantly re-allocating
    ncell_::Array{myint,1} # ncell_[i] = no of particles in cell "i"
    index_::Array{myint,1} # just cumsum of ncell_, really
    Xref_::Array{myint,1}
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
        zeros(myint,ncell),
        zeros(myint,ncell),
        zeros(myint,npart)
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
    Base.Threads.@threads for p in 1:N
        x = P.x_[p]
        y = P.y_[p]
        z = P.z_[p]
        i = sum(map(q->q<=x,m.x_))
        #@assert 0<i<=m.nx
        j = sum(map(q->q<=y,m.y_))
        #@assert 0<j<=m.ny
        k = sum(map(q->q<=z,m.z_))
        #@assert 0<k<=m.nz
        cx_[p] = ix(i,j,k)
    end
    """ (2) count particles in each cell """
    sD.ncell_[:] .= 0 # zero it out before starting the count
    @simd for p in 1:N
        sD.ncell_[cx_[p]] += 1
    end
    nc=sD.ncell_
    print("sD.ncell_: $nc\n")
    s = sum(nc)
    print("sum: $s\n")
    """ (3) build index list (cumsum) """
    sD.index_[1] = 1
    sD.index_[2:end] = 1 .+ cumsum(sD.ncell_)[1:end-1]
    cs=sD.index_
    print("sD.index_: $cs\n")
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

""" Will move this eventually...."""
function main()
    M = rings.mesh(3,4,5,2.0,2.0,2.0)
    P  = rings.particles(4321,1,1,1,1.5,M)
    sD = Main.rings.sortData(P,M)
    rings.sorter!( sD, P, M )
    #
    # Now, to set timestep, need to estimate max collision freq.
end
end
