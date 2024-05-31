include("../CtlrParFiles/IrosActuatorLimits.jl")
include("../CtlrParFiles/IrosPitchPredPars.jl")

struct MagicNumPitchVal
    # Mechanism information
    is_jaw_fixed::Bool
    body_names::Vector{String}
    dof_names::Vector{String}
    actuator_limits::ActuatorLimits
    # Simulation information
    do_scale_traj::Bool
    max_traj_scaling::Float64
    duration_after_traj::Float64
    # Controller information
    add_noise::Bool
    pitch_pred_pars::PitchPredPars
end

function create_magic_num_pitch_val()
    # ----------------------------------------------------------
    # mechanism information
    # ----------------------------------------------------------
    is_jaw_fixed = false
    body_names = ["vehicle", "shoulder", "upperarm", "elbow", "wrist"]
    dof_names = ["roll", "pitch", "yaw", "x", "y", "z", "base", "shoulder", "elbow", "wrist"]
    actuator_limits = create_iros_actuator_limits()

    # ----------------------------------------------------------
    # Simulation information
    # ----------------------------------------------------------
    do_scale_traj = true        # Scale the trajectory?
    max_traj_scaling = 2        # maximum factor to scale trajectory duration by
    duration_after_traj = 1.0   # How long to simulate after trajectory has ended

    # ----------------------------------------------------------
    # Controller information
    # ----------------------------------------------------------
    add_noise = false
    pitch_pred_pars = create_iros_pitch_pred_pars()
    MagicNumPitchVal(
        # Mechanism info
        is_jaw_fixed, body_names, dof_names, actuator_limits,
        # Sim info
        do_scale_traj, max_traj_scaling, duration_after_traj,
        # Controller info
        add_noise, pitch_pred_pars
    )
end