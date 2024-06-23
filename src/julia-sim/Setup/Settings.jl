mutable struct TrajectorySettings
    do_scale::Bool
    max_scaling::Float64
    duration_after::Float64
    max_duration::Float64
end

mutable struct PlottingSettings
    frequency::Int
    factor::Int
    sample_rate::Int
end

mutable struct SimulationSettings
    trajectory::TrajectorySettings
    plotting::PlottingSettings
    num_iterations::Int
    start_visualizer::Bool
end
