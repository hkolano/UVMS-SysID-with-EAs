"""This file is attempt #1 at using an EA to learn the system id 
parameters based on just one or a few trajectories"""

# First just trying to call Julia to see if we can call it from python

from pathlib import Path
from julia import Main

# Trying to set this up so that it can do everything that MainPitchValidation.jl does but in a single python script


def evalConfig(individual, trajectory_names):
    """This function produces an evaluation of the specific individual based on the given trajectories

    The individual is a dictionary full of configuration parameters for the simulator dynamics. The
    trajectories are specified by name. Valid names are hard-coded according to what was collected
    at Hinsdale. The output is a tuple of overall mean-square error
    
    Given an individual (configuration dictionary with specific dynamic parameters), and a set of trajectories to run on, this
    func
    """

    # ----------------------------------------------------------
    #                     Import Libraries
    # ----------------------------------------------------------

    Main.eval(' \n\
    \n\
    using Pkg\n\
    Pkg.activate(".")\n\
    using RigidBodyDynamics, Rotations \n\
    using LinearAlgebra, StaticArrays, DataStructures \n\
    using MeshCat, MeshCatMechanisms, MechanismGeometries \n\
    using CoordinateTransformations \n\
    using GeometryBasics \n\
    using Printf, Plots, CSV, Tables, ProgressBars, Revise \n\
    using Random \n\
    \n\
    using DataFrames, StatsPlots, Interpolations \n\
    \n\
    include("HydroCalc.jl")\n\
    include("SimWExt.jl")\n\
    include("PIDCtlr.jl")\n\
    include("UVMSPlotting.jl")\n\
    include("HelperFuncs.jl")\n\
    include("Noiser.jl")\n\
    \n\
    include("UVMSsetup.jl")\n\
    include("ConfigFiles/MagicNumPitchVal.jl")\n\
    include("ConfigFiles/ConstMagicNums.jl")\n\
    include("ConfigFiles/MagicNumBlueROVHardware.jl")\n\
    include("ConfigFiles/MagicNumAlpha.jl")\n\
    \n\
    trajparsingfile = joinpath("..", "hinsdale_post_processing", "gettrajparamsfromyaml.jl") \n\
    interpolationfile = joinpath("..", "hinsdale_post_processing", "mocap_interpolation.jl") \n\
    simhelperfuncsfile = joinpath("..", "hinsdale_post_processing", "simcomparisonfuncs.jl") \n\
    include(trajparsingfile) \n\
    include(interpolationfile) \n\
    \n\
    urdf_file = joinpath("..","..","urdf", "blue_rov_hardware_fixedjaw_pythonhack.urdf") \n\
    \n\
    \n')

    # ----------------------------------------------------------
    #                 One-Time Mechanism Setup
    # ----------------------------------------------------------
    # Setting this up right now as using just one trajectory to run this quickly

    Main.eval('\n\
    mech_blue_alpha, mvis, joint_dict, body_dict = mechanism_reference_setup(urdf_file)\n\
    include("TrajGenJoints.jl")\n\
    include(simhelperfuncsfile)\n\
    \n\
    \n\
    cob_frame_dict, com_frame_dict = setup_frames(body_dict, body_names, cob_vec_dict, com_vec_dict)\n\
    buoyancy_force_dict, gravity_force_dict = setup_buoyancy_and_gravity(buoyancy_mag_dict, grav_mag_dict)\n\
    \n\
    state = MechanismState(mech_blue_alpha)\n\
    num_dofs = num_velocities(mech_blue_alpha)\n\
    num_actuated_dofs = num_dofs-2\n\
    \n\
    \n\
    all_traj_codes = ["_alt_001-0"]\n\
    # ')
    # all_traj_codes is where I can add more of the trajectories to use for learning

    # ----------------------------------------------------------
    #                 Get Data for Comparison
    # ----------------------------------------------------------

    Main.eval("""\n\
    for (i, trial_code) in enumerate(all_traj_codes)\n\
    \n\
        println("This trial code: $(trial_code)")\n\
    \n\
        # sim_offset = 1\n\
        params, des_df, sim_offset = gettrajparamsfromyaml(trial_code, "otherhome", true)\n\
        # Get mocap data \n\
        mocap_df = get_vehicle_response_from_csv(trial_code, "hinsdale-data-2023", false, true)\n\
        imu_df = get_imu_data_from_csv(trial_code, "hinsdale-data-2023", true)\n\
        imu_df = calc_rpy(imu_df)\n\
        js_df = get_js_data_from_csv(trial_code, "hinsdale-data-2023", "fullrange2", true)\n\
        init_vs, init_vehpose = get_initial_vehicle_velocities(0, mocap_df)\n\
        init_quat, init_ωs = get_initial_conditions(0, imu_df)\n\
    \n\ 
        # ----------------------------------------------------------\n\
        #                         Simulate\n\
        # ----------------------------------------------------------\n\
    \n\
    \n\
        include("HydroCalc.jl")\n\
        include("PIDCtlr.jl")\n\
    \n\
        # Give the vehicle initial conditions from the mocap\n\
        zero!(state)\n\
        set_configuration!(state, joint_dict["vehicle"], [init_quat..., init_vehpose...])\n\
        set_configuration!(state, joint_dict["base"], js_df[1,:axis_e_pos]-3.07)\n\
        set_configuration!(state, joint_dict["shoulder"], js_df[1,:axis_d_pos])\n\
        set_configuration!(state, joint_dict["elbow"], js_df[1,:axis_c_pos])\n\
        set_configuration!(state, joint_dict["wrist"], js_df[1,:axis_b_pos]-2.879)\n\
        init_vs_vector = FreeVector3D(root_frame(mech_blue_alpha), init_vs)\n\
        body_frame_init_vs = RigidBodyDynamics.transform(state, init_vs_vector, default_frame(body_dict["vehicle"]))\n\
        set_velocity!(state, joint_dict["vehicle"], [0, 0, 0, body_frame_init_vs.v...])\n\
    \n\
        # set_configuration!(state, joint_dict["vehicle"], [.9239, 0, 0, 0.382, 0.5, 0., 0.])\n\
        # Start up the controller\n\
        noise_cache = NoiseCache(state)\n\
        filter_cache = FilterCache(state)\n\
        ctlr_cache = CtlrCache(state, noise_cache, filter_cache)\n\
    \n\
        start_buffer = sim_offset+10\n\
        end_buffer = rand(5:0.01:10)\n\
        delayed_params = delayedQuinticTrajParams(params,start_buffer, params.T+start_buffer)\n\
    \n\
        # Simulate the trajectory\n\
        global ts, qs, vs = simulate_with_ext_forces(state, params.T+start_buffer+end_buffer, delayed_params, ctlr_cache, hydro_calc!, pid_control!; Δt=Δt)\n\
    \n\
        @show vs[end]'\n\
        \n\
    \n\
        # ----------------------------------------------------------\n\
        #                      Prepare Plots (just what is necessary for csv generation)\n\
        # ----------------------------------------------------------\n\
        include("UVMSPlotting.jl")\n\
        gr(size=(800, 800)) \n\
        @show sim_offset\n\
    \n\
        sim_palette = palette([:deepskyblue2, :magenta], 4)\n\
        actual_palette = palette([:goldenrod1, :springgreen3], 4)\n\
        \n\
        # Downsample the time steps to goal_freq\n\
        ts_down = [ts[i] for i in 1:sample_rate:length(ts)]\n\
        ts_down_no_zero = ts_down[2:end]\n\
    \n\
        # # Set up data collection dicts\n\
        paths = prep_actual_vels_and_qs_for_plotting(ts_down_no_zero)\n\
        sim_df = DataFrame(paths)\n\
        sim_df[!,"time_secs"] = ts_down_no_zero\n\
        # meas_paths = prep_measured_vels_and_qs_for_plotting()\n\
        # filt_paths = prep_filtered_vels_for_plotting()\n\
    \n\
        deleteat!(sim_df, findall(<(10), sim_df[!,:time_secs]))\n\
        sim_df[!,:time_secs] = sim_df[!,:time_secs] .- minimum(sim_df[!,:time_secs])\n\
    \n\
        p_zed = new_plot()\n\
        @df mocap_df plot!(p_zed, :time_secs, [:z_pose, :y_pose, :x_pose]; :goldenrod1, linewidth=2, label=["mocap z" "mocap_y" "mocap_x"])\n\
        # @df mocap_df plot!(p_zed, :time_secs[1:2500], [:z_pose[1:2500], :y_pose[1:2500], :x_pose[1:2500]]; :goldenrod1, linewidth=2, label=["mocap z" "mocap_y" "mocap_x"])\n\
        @df sim_df plot!(p_zed, :time_secs, [:qs6, :qs5, :qs4]; :deepskyblue2, linewidth=2, linestyle=:dash, label=["sim z" "sim y" "sim x"])\n\
        title!(p_zed, "Vehicle Position for trial "*trial_code)\n\
        ylabel!(p_zed, "Position (m)")\n\
        plot!(p_zed, legend=:outerbottomright)\n\
        label=["actual x_ori" "actual y_ori" "actual z_ori" "actual w_ori"]\n\
        xaxis!(p_zed, grid = (:x, :solid, .75, .9), minorgrid = (:x, :dot, .5, .5))\n\
    \n\
        artificial_offset = 0\n\
    \n\
        p_vehrp = new_plot()\n\
        @df mocap_df plot!(p_vehrp, :time_secs, [:roll, :pitch], palette=actual_palette, linewidth=2, label=["mocap roll" "mocap pitch"])\n\
        xaxis!(p_vehrp, grid = (:x, :solid, .75, .9), minorgrid = (:x, :dot, .5, .5))\n\
        @df sim_df plot!(p_vehrp, :time_secs.+artificial_offset, [:qs1, :qs2], \n\
            palette=sim_palette, linewidth=2, linestyle=:dash, \n\
            label=["sim roll" "sim pitch"])\n\
            plot!(p_vehrp, legend=:outerbottomright)\n\
        @df imu_df plot!(p_vehrp, :time_secs, [:roll, :pitch], linewidth=2, label=["imu roll" "imu pitch"])\n\
        ylabel!("Vehicle Orientation (rad)")\n\
        title!("BlueROV Orientation")\n\
    \n\
        @show rad2deg(get_pitch_rmse(imu_df, sim_df))\n\
        @show rad2deg(get_pitch_rmse(imu_df, sim_df, true, -1.))\n\
    \n\
        p_js = new_plot()\n\
        @df js_df plot!(p_js, :time_secs, cols(3:6); palette=actual_palette, linewidth=2)\n\
        xaxis!(p_js, grid = (:x, :solid, .75, .9), minorgrid = (:x, :dot, .5, .5))\n\
        # @df des_df plot!(p_js, :time_secs, cols(2:5); palette=:grayC, linewidth=2, linestyle=:dash)\n\
        @df sim_df plot!(p_js, :time_secs, \n\
            [cols(7).+3.07, cols(8), cols(9), cols(10).+2.879]; #, cols(11)]; \n\
            palette=sim_palette, linewidth=2, linestyle=:dash, \n\
            label=["sim axis e" "sim axis d" "sim axis c" "sim axis b"])\n\
        plot!(p_js, legend=:outerbottomright)\n\
        ylabel!("Joint position (rad)")\n\
        title!("Alpha Arm Joint Positions")\n\
        plot!(p_js, ylims=(-.5, 6))\n\
    \n\
        super_plot = plot(p_js, p_vehrp, layout=(2, 1), plot_title="Sim vs Hinsdale, traj "*trial_code*" (artificial offset "*string(artificial_offset)*"s)")\n\
        display(super_plot)\n\
        \n\
        # ----------------------------------------------------------\n\
        #                 Save Trajectory to CSV\n\
        # ----------------------------------------------------------\n\
        # only save the trajectory if Joint 1 doesn't exceed the joint velocity limits (which is a proxy for indicating whether it is unstable)\n\
    \n\
        # Rows:\n\
        # 1-10: Actual position data (qs)\n\
        # 11-20: Actual velocity data (vs)\n\
        # 21-30: Noisy position data (noisy_qs)\n\
        # 31-40: Noisy velocity data (noisy_vs)\n\
        # 41-44: Desired velocities \n\
        deleteat!(sim_df, 1:2:length(sim_df[!,:time_secs]))\n\
        const_dt_imu_df = interp_at_timesteps(sim_df[!,:time_secs], imu_df, [:roll, :pitch])\n\
        const_dt_js_df = interp_at_timesteps(sim_df[!,:time_secs], js_df, names(js_df))\n\
        combo_df = hcat(sim_df, const_dt_imu_df[!,[:roll,:pitch]], const_dt_js_df[:,2:end])\n\
        select!(combo_df, Not([:w_ori, :x_ori, :y_ori, :z_ori]))\n\
        new_file_name = joinpath("..","..","data", "sim_trajs", trial_code*".csv")\n\
        CSV.write(new_file_name, combo_df)\n\
    end\n\
    """)
