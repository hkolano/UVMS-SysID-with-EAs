mutable struct WorldParameters
    rho::Float64
    Δt::Float64
end

mutable struct ActuatorLimits
    joints::Dict{String, SVector{2, Float64}}
    velocities::Dict{String, SVector{2, Float64}}
    torques::Dict{String, Float64}
    d_taus::Dict{String, Float64}
end

mutable struct BuoyancyParameters
    centers_of_buoyancy::Dict{String, SVector{3, Float64}}
    magnitudes::Dict{String, Float64}
    forces::Dict{String, FreeVector3D}
end

mutable struct GravityParameters
    centers_of_mass::Dict{String, SVector{3, Float64}}
    magnitudes::Dict{String, Float64}
    forces::Dict{String, FreeVector3D}
end

mutable struct DragParameters
    links::Dict{String, SVector{3, Float64}}
    linear_coefficients::Vector{Float64}
    nonlinear_coefficients::Vector{Float64}
end

mutable struct NoiseParameters
    v_angular_velocity::Distributions.Normal{Float64}
    arm_position::Distributions.Normal{Float64}
    acceleration::Distributions.Normal{Float64}
    gyroscope_random_walk::Distributions.Normal{Float64}
    acceleration_random_walk::Distributions.Normal{Float64}
end

mutable struct ModelParameters
    is_jaw_fixed::Bool
    body_names::Vector{String}
    degrees_of_freedom_names::Vector{String}
    actuator_limits::ActuatorLimits
    buoyancy::BuoyancyParameters
    gravity::GravityParameters
    vehicle_extras::Vector{String}
    drag::DragParameters
    noise::NoiseParameters
end

mutable struct ControllerParameters
    add_noise::Bool
    Kp::Dict{String, Float64}
    Ki::Dict{String, Float64}
    Kd::Dict{String, Float64}
    do_feedforward::Bool
    feedforward_propogation::Float64
    filtering_kernel::Int
    frequency::Int
    num_steps::Int
end

mutable struct SimulationParameters
    world::WorldParameters
    model::ModelParameters
    controller::ControllerParameters
end
