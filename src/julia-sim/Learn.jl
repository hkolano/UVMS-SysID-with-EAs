#=
Experimental flight code for using an evolutionary algorithm to learn
    the dynamic parameters of the BlueROV based on trajectory data
=#

# ----------------------------------------------------------
#                     Import Libraries
# ----------------------------------------------------------
#%%
using RigidBodyDynamics, Rotations
using LinearAlgebra, StaticArrays, DataStructures
using MeshCat, MeshCatMechanisms, MechanismGeometries
using CoordinateTransformations
using GeometryBasics
using Printf, Plots, CSV, Tables, ProgressBars, Revise
using Random
using YAML

using DataFrames, StatsPlots, Interpolations

include("HydroCalc.jl")
include("SimWExt.jl")
include("PIDCtlr.jl")
include("UVMSPlotting.jl")
include("HelperFuncs.jl")
include("Noiser.jl")
include("TrajGenJoints.jl")
include("UVMSsetup.jl")

include("Setup/SetupParameters.jl")
include("Setup/SetupSettings.jl")

trajparsingfile = joinpath("..", "hinsdale_post_processing", "gettrajparamsfromyaml.jl")
interpolationfile = joinpath("..", "hinsdale_post_processing", "mocap_interpolation.jl")
simhelperfuncsfile = joinpath("..", "hinsdale_post_processing", "simcomparisonfuncs.jl")
include(trajparsingfile)
include(interpolationfile)
include(simhelperfuncsfile)

# ----------------------------------------------------------
#      Load in our simulation settings and parameters
# ----------------------------------------------------------
# Load in parameters and settings
parameters = setup_parameters()
settings = setup_settings(parameters.controller.frequency, parameters.world.Δt)

# These should not change regardless of dynamic parameters
urdf_file = joinpath("urdf", "blue_rov_hardware_fixedjaw.urdf")
uvms_mechanism = mechanism_reference_setup(
    urdf_file,
    parameters.model.body_names,
    parameters.model.degrees_of_freedom_names,
    settings.start_visualizer
)
traj_gen_joints = create_traj_gen_joints(uvms_mechanism.mechanism)

# These should also not change regardless of dynamic parameters
state = MechanismState(uvms_mechanism.mechanism)
num_dofs = num_velocities(uvms_mechanism.mechanism)
num_actuated_dofs = num_dofs-2

all_traj_codes = ["_alt_001-0"]
num_trajectories = length(all_traj_codes)

#%%
# ----------------------------------------------------------
#                 Get Data for Comparison
# ----------------------------------------------------------
# trial_code = "004-0"
# trial_code = "baseline1"

# Let's first aggregate all of the data we are going to use for learning
# and put it in memory so we don't have to keep loading it in
struct TrajInfo
    params::quinticTrajParams
    des_df::DataFrame
    sim_offset::Float64
    mocap_df::DataFrame
    imu_df::DataFrame
    js_df::DataFrame
    init_vs::Vector{Float64}
    init_vehpose::Vector{Float64}
    init_quat::Vector{Float64}
    init_ws::Matrix{Float64}
end

#%%
traj_infos = Vector{TrajInfo}()
for (i, trial_code) in enumerate(all_traj_codes)
    println("Loading in data for this trial code: $(trial_code)")

    # Get quintic parameters, desired trajectory?, simulation offsetvpn.oregonstate.edu
    params, des_df, sim_offset = gettrajparamsfromyaml(traj_gen_joints, parameters.model.degrees_of_freedom_names, trial_code, "otherhome")

    # Collect mocap, imu, and joint data
    mocap_df = get_vehicle_response_from_csv(trial_code, "hinsdale-data-2023", false)
    imu_df = get_imu_data_from_csv(trial_code, "hinsdale-data-2023")
    imu_df = calc_rpy(imu_df)
    js_df = get_js_data_from_csv(trial_code, "hinsdale-data-2023")

    init_vs, init_vehpose = get_initial_vehicle_velocities(0, mocap_df)
    init_quat, init_ws = get_initial_conditions(0, imu_df)

    # Store this all for later
    traj_infos = push!(traj_infos, TrajInfo(
        params, des_df, sim_offset, mocap_df, imu_df, js_df, init_vs, init_vehpose, init_quat, init_ws
    ))
end

#%%

# Load in dynamic parameters from yaml file

config_dir = joinpath("src", "julia-sim", "ConfigFiles", "DynamicParameters.yaml")
dynamic_parameters = YAML.load_file(config_dir)

parameters.model.buoyancy.centers_of_buoyancy = Dict{String, SVector{3,Float64}}(dynamic_parameters["cob_vec_dict"])
parameters.model.buoyancy.magnitudes = Dict{String, Float64}(dynamic_parameters["buoyancy_mag_dict"])
link_volumes = dynamic_parameters["link_volumes"]
for (k,v) in dynamic_parameters["link_volumes"]
    parameters.model.buoyancy.magnitudes[k] = v*parameters.world.rho*9.81*0.001
end

parameters.model.gravity.centers_of_mass = Dict{String, SVector{3, Float64}}(dynamic_parameters["com_vec_dict"])
parameters.model.gravity.magnitudes = Dict{String, Float64}(dynamic_parameters["grav_mag_dict"])
link_masses = dynamic_parameters["link_masses"]
for (k,v) in link_masses
    parameters.model.gravity.magnitudes[k] = v*9.81
end

d_lin_angular = dynamic_parameters["d_lin_angular"]
parameters.model.drag.linear_coefficients = Vector{Float64}(dynamic_parameters["d_lin_coeffs"])
push!(parameters.model.drag.linear_coefficients, d_lin_angular, d_lin_angular, d_lin_angular)

d_nonlin_angular = dynamic_parameters["d_nonlin_angular"]
parameters.model.drag.nonlinear_coefficients = dynamic_parameters["d_nonlin_coeffs"]
push!(parameters.model.drag.nonlinear_coefficients, d_nonlin_angular, d_nonlin_angular, d_nonlin_angular)

# These will change depending on dynamic parameters
parameters.model.buoyancy.centers_of_buoyancy, model.gravity.centers_of_mass = setup_frames(
    uvms_mechanism.bodies,
    parameters.model.body_names,
    parameters.model.buoyancy.centers_of_buoyancy,
    parameters.model.gravity.centers_of_mass,
    parameters.model.vehicle_extras,
    uvms_mechanism.visualizer
)
setup_buoyancy_and_gravity(
    uvms_mechanism.mechanism,
    parameters.model.buoyancy.forces,
    parameters.model.gravity.forces,
    parameters.model.buoyancy.magnitudes,
    parameters.model.gravity.magnitudes
)

#%%
# Some other variables I missed
sensor_noise_dist = create_sensor_noise_dist()

#%%

struct SimResult
    ts::Vector{Float64}
    qs::Vector{SegmentedVector{JointID, Float64, Base.OneTo{JointID}, Vector{Float64}}}
    vs::Vector{SegmentedVector{JointID, Float64, Base.OneTo{JointID}, Vector{Float64}}}
end

#%%
save_to_csv = false
show_plots = false
show_animation = false
#%%

sim_results = Vector{SimResult}()
for (i, (traj_info, trial_code)) in enumerate(zip(traj_infos, all_traj_codes))
    # ----------------------------------------------------------
    #                         Simulate
    # ----------------------------------------------------------
    println("Simulating trial $(trial_code)")

    # Give the vehicle initial conditions from the mocap
    zero!(state)
    set_configuration!(state, joint_dict["vehicle"], [traj_info.init_quat..., traj_info.init_vehpose...])
    set_configuration!(state, joint_dict["base"], traj_info.js_df[1,:axis_e_pos]-3.07)
    set_configuration!(state, joint_dict["shoulder"], traj_info.js_df[1,:axis_d_pos])
    set_configuration!(state, joint_dict["elbow"], traj_info.js_df[1,:axis_c_pos])
    set_configuration!(state, joint_dict["wrist"], traj_info.js_df[1,:axis_b_pos]-2.879)
    init_vs_vector = FreeVector3D(root_frame(mech_blue_alpha), traj_info.init_vs)
    body_frame_init_vs = RigidBodyDynamics.transform(state, init_vs_vector, default_frame(body_dict["vehicle"]))
    set_velocity!(state, joint_dict["vehicle"], [0, 0, 0, body_frame_init_vs.v...])

    # Start up the controller
    noise_cache = NoiseCache(state)
    filter_cache = FilterCache(state)
    ctlr_cache = CtlrCache(state, noise_cache, filter_cache, magic_num_pitch_val.dof_names)

    start_buffer = traj_info.sim_offset+10
    end_buffer = 10
    delayed_params = delayedQuinticTrajParams(traj_info.params,start_buffer, traj_info.params.T+start_buffer)

    # Simulate the trajectory
    global ts, qs, vs = simulate_with_ext_forces(
        actuator_limits,
        pitch_pred_pars,
        sensor_noise_dist,
        num_dofs,
        com_frame_dict,
        cob_frame_dict,
        const_magic_nums,
        magic_num_pitch_val,
        magic_num_bluerov_with_alpha_arm,
        body_dict,
        joint_dict,
        state,
        traj_info.params.T+start_buffer+end_buffer,
        delayed_params,
        ctlr_cache,
        hydro_calc!,
        mvis,
        pid_control!
    )
    # Store this all for later
    sim_results = push!(sim_results, SimResult(
        ts, qs, vs
    ))

end

#%%
for (i, (sim_result, traj_info, trial_code)) in enumerate(zip(sim_results, traj_infos, all_traj_codes))
    # ----------------------------------------------------------
    #                      Prepare Plots
    # ----------------------------------------------------------
    gr(size=(800, 800))
    @show traj_info.sim_offset

    sim_palette = palette([:deepskyblue2, :magenta], 4)
    actual_palette = palette([:goldenrod1, :springgreen3], 4)

    # Downsample the time steps to goal_freq
    ts_down = [ts[i] for i in 1:const_magic_nums.sample_rate:length(ts)]
    ts_down_no_zero = ts_down[2:end]

    # # Set up data collection dicts
    paths = prep_actual_vels_and_qs_for_plotting(ts_down_no_zero, const_magic_nums)
    sim_df = DataFrame(paths)
    sim_df[!,"time_secs"] = ts_down_no_zero
    # meas_paths = prep_measured_vels_and_qs_for_plotting()
    # filt_paths = prep_filtered_vels_for_plotting()

    deleteat!(sim_df, findall(<(10), sim_df[!,:time_secs]))
    sim_df[!,:time_secs] = sim_df[!,:time_secs] .- minimum(sim_df[!,:time_secs])

    p_zed = new_plot()
    @df traj_info.mocap_df plot!(p_zed, :time_secs, [:z_pose, :y_pose, :x_pose]; :goldenrod1, linewidth=2, label=["mocap z" "mocap_y" "mocap_x"])
    @df sim_df plot!(p_zed, :time_secs, [:qs6, :qs5, :qs4]; :deepskyblue2, linewidth=2, linestyle=:dash, label=["sim z" "sim y" "sim x"])
    title!(p_zed, "Vehicle Position for trial "*trial_code)
    ylabel!(p_zed, "Position (m)")
    plot!(p_zed, legend=:outerbottomright)
    label=["actual x_ori" "actual y_ori" "actual z_ori" "actual w_ori"]
    xaxis!(p_zed, grid = (:x, :solid, .75, .9), minorgrid = (:x, :dot, .5, .5))

    # @df mocap_df plot!(p_zed, :time_secs[1:2500], [:z_pose[1:2500], :y_pose[1:2500], :x_pose[1:2500]]; :goldenrod1, linewidth=2, label=["mocap z" "mocap_y" "mocap_x"])
    artificial_offset = 0

    p_vehrp = new_plot()
@df traj_info.mocap_df plot!(p_vehrp, :time_secs, [:roll, :pitch], palette=actual_palette, linewidth=2, label=["mocap roll" "mocap pitch"])
    xaxis!(p_vehrp, grid = (:x, :solid, .75, .9), minorgrid = (:x, :dot, .5, .5))
    @df sim_df plot!(p_vehrp, :time_secs.+artificial_offset, [:qs1, :qs2],
        palette=sim_palette, linewidth=2, linestyle=:dash,
        label=["sim roll" "sim pitch"])
        plot!(p_vehrp, legend=:outerbottomright)
    @df traj_info.imu_df plot!(p_vehrp, :time_secs, [:roll, :pitch], linewidth=2, label=["imu roll" "imu pitch"])
    ylabel!("Vehicle Orientation (rad)")
    title!("BlueROV Orientation")

    @show rad2deg(get_pitch_rmse(traj_info.imu_df, sim_df))
    @show rad2deg(get_pitch_rmse(traj_info.imu_df, sim_df, true, -1.))

    # super_ori_plot = plot(p_zed, p_vehrp, layout=(2,1), plot_title="Comparison for "*trial_code)

    p_js = new_plot()
    @df traj_info.js_df plot!(p_js, :time_secs, cols(3:6); palette=actual_palette, linewidth=2)
    xaxis!(p_js, grid = (:x, :solid, .75, .9), minorgrid = (:x, :dot, .5, .5))
    # @df des_df plot!(p_js, :time_secs, cols(2:5); palette=:grayC, linewidth=2, linestyle=:dash)
    @df sim_df plot!(p_js, :time_secs,
        [cols(7).+3.07, cols(8), cols(9), cols(10).+2.879]; #, cols(11)];
        palette=sim_palette, linewidth=2, linestyle=:dash,
        label=["sim axis e" "sim axis d" "sim axis c" "sim axis b"])
    plot!(p_js, legend=:outerbottomright)
    ylabel!("Joint position (rad)")
    title!("Alpha Arm Joint Positions")
    plot!(p_js, ylims=(-.5, 6))
    # if bool_plot_velocities == true
    #     plot_des_vs_act_velocities(ts_down_no_zero,
    #         paths, des_paths, meas_paths, filt_paths,
    #         plot_veh=false, plot_arm=true)
    # end

    # if bool_plot_positions == true
    #     plot_des_vs_act_positions(ts_down_no_zero, des_ts,
    #         paths, des_paths, meas_paths,
    #         plot_veh = true, plot_arm=true)
    # end

    # if bool_plot_taus == true
    #     plot_control_taus(ctlr_cache, ts_down)
    # end

    super_plot = plot(p_js, p_vehrp, layout=(2, 1), plot_title="Sim vs Hinsdale, traj "*trial_code*" (artificial offset "*string(artificial_offset)*"s)")
    display(super_plot)
    # ----------------------------------------------------------
    #                  Animate the Trajectory
    # ----------------------------------------------------------
    # Use stop_step to visualize specific points in the trajectory
    # stop_step = 29*1000
    if show_animation == true
        print("Animating... ")
        # MeshCatMechanisms.animate(mvis, ts[1:stop_step], qs[1:stop_step]; realtimerate = 5.0)
        MeshCatMechanisms.animate(mvis, ts, qs; realtimerate=1.)
        println("done.")
    end

    # ----------------------------------------------------------
    #                 Save Trajectory to CSV
    # ----------------------------------------------------------
    # only save the trajectory if Joint 1 doesn't exceed the joint velocity limits (which is a proxy for indicating whether it is unstable)
    if save_to_csv == true
        # Rows:
        # 1-10: Actual position data (qs)
        # 11-20: Actual velocity data (vs)
        # 21-30: Noisy position data (noisy_qs)
        # 31-40: Noisy velocity data (noisy_vs)
        # 41-44: Desired velocities
        deleteat!(sim_df, 1:2:length(sim_df[!,:time_secs]))
        const_dt_imu_df = interp_at_timesteps(sim_df[!,:time_secs], traj_info.imu_df, [:roll, :pitch])
        const_dt_js_df = interp_at_timesteps(sim_df[!,:time_secs], traj_info.js_d, names(traj_info.js_d))
        combo_df = hcat(sim_df, const_dt_imu_df[!,[:roll,:pitch]], const_dt_js_df[:,2:end])
        select!(combo_df, Not([:w_ori, :x_ori, :y_ori, :z_ori]))
        new_file_name = joinpath("data", "sim_trajs", trial_code*".csv")
        CSV.write(new_file_name, combo_df)
    else
        println("Not saving trajectory")
    end
    println("")
end