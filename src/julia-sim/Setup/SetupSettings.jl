include("Settings.jl")

function setup_settings(controller_frequency, Δt)::SimulationSettings
    do_scale = true
    max_scaling = 2
    duration_after = 1.0
    max_duration = 20.
    trajectory_settings = TrajectorySettings(
        do_scale,
        max_scaling,
        duration_after,
        max_duration
    )
    plot_freq = Int(50)      # frequency of the output CSV and plots
    plot_factor = controller_frequency/plot_freq
    sample_rate = Int(floor((1/Δt)/plot_freq))
    @assert rem(plot_factor, 1) == 0.0 "Please pick a plot frequency that is an even divisor of the control frequency."
    plotting_settings = PlottingSettings(
        plot_freq,
        plot_factor,
        sample_rate
    )
    num_iterations = 50
    start_visualizer = false
    return SimulationSettings(
        trajectory_settings,
        plotting_settings,
        num_iterations,
        start_visualizer
    )
end
