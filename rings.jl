module rings

using SpecialFunctions
using Random
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
#Random.seed!(2123408012)

myint = Int32
"""
################################################################################
struct particles:

    Note: Struct-of-Arrays faster than Array-of-Structs

    Note: in action-angle variables for harmonic oscillator (Hamilton-Jacobi), canonical action is
    equal to total energy E = (1/2)mv^2 + (1/2)kz^2. Then
    E = (1/2) m^2 Omega^2 z_max^2  and z_max = sqrt(2*E) / (m * Omega).

"""
struct particles
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
    xlim::NTuple{2,Real}, ylim::NTuple{2,Real}, zlim::NTuple{2,Real},
    dvx::Real, dvy::Real, dvz::Real, shear::Real )
    #
    m_ = (m*ones(N)...,)
    D_ = (D*ones(N)...,)
    x_ = xlim[1] .+ (xlim[2]-xlim[1])*Random.rand(N)
    y_ = ylim[1] .+ (ylim[2]-ylim[1])*Random.rand(N)
    z_ = zlim[1] .+ (zlim[2]-zlim[1])*Random.rand(N)
    vx_= dvx*Random.randn(N)
    vy_= dvy*Random.randn(N) + shear*x_
    vz_= dvz*Random.randn(N)
    Py_= zeros(N)
    Ez_= zeros(N)
    particles(N,m_,D_,x_,y_,z_,vx_,vy_,vz_,Py_,Ez_)
end
"""
################################################################################
struct mesh: an immutable struct for recording the size and location of
    the Cartesian mesh used to sort particles.
"""
struct mesh
    nx::myint # number of CELLS (not nodes) in x-dir
    ny::myint # etc
    nz::myint
    x_::Tuple{Vararg{Float64}} # x-location of cell boundaries (i.e. nodes)
    y_::Tuple{Vararg{Float64}} # Set as tuples, to enforce immutability
    z_::Tuple{Vararg{Float64}}
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
    )
end
""" Convenience 2D (x,z) constructor: """
mesh(nx::Int,nz::Int,Lx::Real,Hz::Real) = mesh(nx,1,nz,Lx,Inf,Hz)

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
sortData(ncell::Int,npart::Int) = sortData(
    M      = ncell,
    N      = npart,
    cx_    = zeros(myint,npart),
    ncell_ = zeros(myint,ncell),
    index_ = zeros(myint,ncell),
    Xref_  = zeros(myint,npart)
)

function sorter!(sD::sortData, P::particles, m::mesh)
    cx_ = sD.cx_ # <-- make sure this is an alias, not a copy (right?)
    nx = m.nx; ny = m.ny; nz = m.nz, N = P.N, M = nx*ny*nz
    @assert sD.M = M
    @assert sD.N = N
    # I think there might be library utility functions available for these two...:
    function ix(i::Int,j::Int,k::Int)
        @assert 0 < i <= nx
        @assert 0 < j <= ny
        @assert 0 < k <= nz
        i + nx*j + nx*ny*k
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
        @assert 0<i<=m.nx
        j = sum(map(q->q<=y,m.y_))
        @assert 0<j<=m.ny
        k = sum(map(q->q<=z,m.z_))
        @assert 0<k<=m.nz
        cx_[dp] = ix(i,j,k)
    end
    """ (2) count particles in each cell """
    sD.ncell_ *= 0 # zero it out before starting the count
    Base.Threads.@threads for p in 1:N
        sD.ncell_[cx_[p]] += 1
    end
    """ (3) build index list (cumsum) """
    sD.index_[:] = cumsum(sD.ncell_)
    """ (4) build cross-reference list """
    temp_ = zeros(myint,M)
    # Base.Threads.@threads
    for p in 1:N
        c = cx_[p]
        k = sD.index_[c] + temp_[c]
        sD.Xref[ k ] = p
        temp_[c]    += 1
    end
end

end
