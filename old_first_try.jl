# Just a comment!
# And another comment.

using Plots
using Random
Random.seed!(12301091)
minute = 60.0
hour = 60 * minute

Omega = 2*π/(9*hour) # Keplerian at B-ring
N = 10000 # number of time-steps
M = 10 # number of particles
# fiducial peculiar velocity about 0.01-0.1 cm/s
Lx = 1.0e3
Ly = 1.0e3 # size of domain

tau = 10.0  # time step

x = rand() * Lx;
y = rand() * Ly

xdot = 0.01 * randn()
ydot = 0.01 * randn() - 1.5*Omega*x

x_ = zeros(N)
y_ = zeros(N)
xdot_ = zeros(N)
ydot_ = zeros(N)

x_[1] = 0.0 #480.0 #x
y_[1] = 400.0 #y
xdot_[1] = 0.01 #xdot
ydot_[1] = 0.0 - 1.5*Omega*x_[1] # ydot - 1.5*Omega*x_[1]

let
t = 0.0
for i in 1:(N-1)
    xdot_[i+1] = xdot_[i] - 0.5 * tau * Omega^2 * x_[i]
    Py = ydot_[i] + 2*Omega*x_[i]
    #
    xdot_[i+1] += tau * Omega * Py
    ydot_[i+1] = Py - Omega*x_[i] -Omega*(x_[i] + tau*xdot_[i+1])
    #
    x_[i+1] = x_[i] + tau * xdot_[i+1]
    #y_[i+1] = mod( y_[i] + tau * ydot_[i+1], Ly);
    y_[i+1] = y_[i] + tau * ydot_[i+1]
    #
    #ydot =
    #
    xdot_[i+1] += tau * Omega * Py
    #
    xdot_[i+1] -= 0.5 * tau * (Omega^2 * x_[i+1])
    #xdot_[i+1] -= 0.5 * tau * (Omega^2 * xtest);
    ydot_[i+1]  = Py - 2 * Omega * x_[i+1]
    #
    #global t += tau
    t += tau
    if (x_[i+1] < -0.5*Lx)
        x_[i+1] += Lx
        y_[i+1] -= mod( 1.5*Omega*Lx*t, Ly)
        ydot_[i+1] -= 1.5 * Omega * Lx
    elseif (x_[i+1] ≥ 0.5*Lx)
        x_[i+1] -= Lx
        y_[i+1] += mod( 1.5*Omega*Lx*t, Ly)
        ydot_[i+1] += 1.5 * Omega * Lx
    end
    y_[i+1] = mod( y_[i+1] + 0.5*Ly, Ly) - 0.5*Ly
end
end



mutable struct Particle
    m::Float64 # mass
    D::Float64 # diameter
    x::Float64 # x-position
    xdot::Float64 # x-speed
    Px::Float64 # x-momentum
    y::Float64
    ydot::Float64
    Py::Float64
end

pt = Particle(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)

function move!( part::Particle , tau::Float64 , t )
    part.xdot += 0.0*xdot - 0.5 * tau * Omega^2 * part.x;
    part.Py = part.ydot + 2*Omega*part.x;
    #
    part.xdot += tau * Omega * part.Py;
    part.ydot = part.Py - Omega * part.x -Omega*(part.x + tau*part.xdot);
    #
    part.x += tau * part.xdot;
    part.y += tau * part.ydot;
    #
    part.xdot += tau * Omega * part.Py;
    #
    part.xdot -= 0.5 * tau * (Omega^2 * part.x);
    #xdot_[i+1] -= 0.5 * tau * (Omega^2 * xtest);
    part.ydot = part.Py - 2*Omega*part.x;
    #
    if (part.x < -0.5*Lx)
        part.x += Lx;
        part.y -= mod( 1.5*Omega*Lx*t, Ly);
        part.ydot -= 1.5 * Omega * Lx;
    elseif (part.x ≥ 0.5*Lx)
        part.x -= Lx;
        part.y += mod( 1.5*Omega*Lx*t, Ly);
        part.ydot += 1.5 * Omega * Lx;
    end
    part.y = mod( part.y,Ly);
end

function collide!(part1::Particle, part2::Particle)
    # This part should be put somehwere else...
    part1.Px = part1.m * part1.xdot
    part2.Px = part2.m * part2.xdot
    part1.Py = part1.m * (part1.ydot + 2*Omega*part1.x)
    part2.Py = part2.m * (part2.ydot + 2*Omega.part2.x)
    #
    Px = part1.Px + part2.Px
    Py = part1.Py + part2.Py
end
