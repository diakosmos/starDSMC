module rings
"""
Note to self re parallelization:

see https://discourse.julialang.org/t/struct-of-arrays-soa-vs-array-of-structs-aos/30015

Quoting:
@inbounds , @simd , @simd ivdep
check if it has been vectorized with @code_llvm.

Using BenchmarkTools
@btime
"""

using SpecialFunctions
"""
################################################################################
struct particles:

    Note: Struct-of-Arrays faster than Array-of-Structs

    Note: in action-angle variables for harmonic oscillator (Hamilton-Jacobi), canonical action is
    equal to total energy E = (1/2)mv^2 + (1/2)kz^2. Then
    E = (1/2) m^2 Omega^2 z_max^2  and z_max = sqrt(2*E) / (m * Omega).

"""
struct particles
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

"""
################################################################################
struct mesh: an immutable struct for recording the size and location of
    the Cartesian mesh used to sort particles.
"""
struct mesh
    nx::Int64 # number of CELLS (not nodes) in x-dir
    ny::Int64 # etc
    nz::Int64
    x_::Tuple{Vararg{Float64}} # x-location of cell boundaries (i.e. nodes)
    y_::Tuple{Vararg{Float64}} # Set as tuples, to enforce immutability
    z_::Tuple{Vararg{Float64}}
end
"""
 Nominal outer constructor. Note Hz is vertical scale height.
"""
function mesh(nx,ny,nz,Lx::Float64,Ly::Float64,Hz::Float64)
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
mesh(nx::Int64,nz::Int64,Lx::Float64,Hz::Float64) = mesh(nx,1,nz,Lx,Inf,Hz)

"""
################################################################################
struct sortData:
"""
struct sortData
    
end

end
