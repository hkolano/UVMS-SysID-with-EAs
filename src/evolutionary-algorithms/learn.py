"""This block just sets up and runs the julia sim to get the variables we're interested in tracking"""

# First just trying to call Julia to see if we can call it from python
from pathlib import Path
import os
from julia.api import Julia
# Julia(runtime=os.path.expanduser("~")+'/julia-1.8.2/bin/julia-1.8.2',compiled_modules=False)
Julia(compiled_modules=False)
from julia import Main
import numpy as np

def individualListToDict(individual_list_format):
    """This takes in an individual as a list of parameters and turns it into a dictionary for evaluation"""
    individual_dict_format = {
        "cob_vec_dict": {
            "vehicle": individual_list_format[0:3],
            "foamL": individual_list_format[3:6],
            "foamR": individual_list_format[6:9],
            "shoulder": individual_list_format[9:12],
            "upperarm": individual_list_format[12:15],
            "elbow": individual_list_format[15:18],
            "wrist": individual_list_format[18:21],
            "jaw": individual_list_format[21:24]
        },
        "buoyancy_mag_dict": {
            "vehicle": individual_list_format[24],
            "foamL": individual_list_format[25],
            "foamR": individual_list_format[26],
            "shoulder": individual_list_format[27],
            "upperarm": individual_list_format[28],
            "armbase": individual_list_format[29],
            "jaw": individual_list_format[30],
            "wrist": individual_list_format[31],
            "elbow": individual_list_format[32]
        },
        "com_vec_dict": {
            "vehicle": individual_list_format[33:36],
            "weightCA": individual_list_format[36:39],
            "weightBL": individual_list_format[39:42],
            "weightBR": individual_list_format[42:45],
            "dvl": individual_list_format[45:48],
            "dvlbracket": individual_list_format[48:51],
            "armbase": individual_list_format[51:54],
            "shoulder": individual_list_format[54:57],
            "upperarm": individual_list_format[57:60],
            "elbow": individual_list_format[60:63],
            "wrist": individual_list_format[63:66],
            "jaw": individual_list_format[66:69],
            "jaw_wrt_wrist": individual_list_format[69:72]
        },
        "grav_mag_dict": {
            "vehicle": individual_list_format[72],
            "weightCA": individual_list_format[73],
            "weightBL": individual_list_format[74],
            "weightBR": individual_list_format[75],
            "dvl": individual_list_format[76],
            "dvlbracket": individual_list_format[77],
            "armbase": individual_list_format[78],
            "shoulder": individual_list_format[79],
            "upperarm": individual_list_format[80],
            "elbow": individual_list_format[81],
            "jaw": individual_list_format[82],
            "wrist": individual_list_format[83]
        },
        "drag": {
            "d_lin_angular": individual_list_format[84],
            "d_nonlin_angular": individual_list_format[85],
            "d_lin_coeffs": individual_list_format[86:89],
            "d_nonlin_coeffs": individual_list_format[89:92]
        },
        "link_volumes": {
            "shoulder": individual_list_format[92],
            "upperarm": individual_list_format[93],
            "elbow": individual_list_format[94],
            "wrist": individual_list_format[95],
            "armbase": individual_list_format[96],
            "jaw": individual_list_format[97]
        },
        "link_masses": {
            "shoulder": individual_list_format[98],
            "upperarm": individual_list_format[99],
            "elbow": individual_list_format[100],
            "wrist": individual_list_format[101],
            "armbase": individual_list_format[102],
            "jaw": individual_list_format[103]
        },
        "link_drags": {
            "shoulder": individual_list_format[104:107],
            "upperarm": individual_list_format[107:110],
            "elbow": individual_list_format[110:113],
            "wrist": individual_list_format[113:116],
            "jaw": individual_list_format[116:119]
        }
    }
    return individual_dict_format

def individualDictToList(individual_dict_format):
    individual_list_format = \
        individual_dict_format["cob_vec_dict"]["vehicle"] + \
        individual_dict_format["cob_vec_dict"]["foamL"] + \
        individual_dict_format["cob_vec_dict"]["foamR"] + \
        individual_dict_format["cob_vec_dict"]["shoulder"] + \
        individual_dict_format["cob_vec_dict"]["upperarm"] + \
        individual_dict_format["cob_vec_dict"]["elbow"] + \
        individual_dict_format["cob_vec_dict"]["wrist"] + \
        individual_dict_format["cob_vec_dict"]["jaw"] + \
        [individual_dict_format["buoyancy_mag_dict"]["vehicle"]] + \
        [individual_dict_format["buoyancy_mag_dict"]["foamL"]] + \
        [individual_dict_format["buoyancy_mag_dict"]["foamR"]] + \
        [individual_dict_format["buoyancy_mag_dict"]["shoulder"]] + \
        [individual_dict_format["buoyancy_mag_dict"]["upperarm"]] + \
        [individual_dict_format["buoyancy_mag_dict"]["armbase"]] + \
        [individual_dict_format["buoyancy_mag_dict"]["jaw"]] + \
        [individual_dict_format["buoyancy_mag_dict"]["wrist"]] + \
        [individual_dict_format["buoyancy_mag_dict"]["elbow"]] + \
        individual_dict_format["com_vec_dict"]["vehicle"] + \
        individual_dict_format["com_vec_dict"]["weightCA"] + \
        individual_dict_format["com_vec_dict"]["weightBL"] + \
        individual_dict_format["com_vec_dict"]["weightBR"] + \
        individual_dict_format["com_vec_dict"]["dvl"] + \
        individual_dict_format["com_vec_dict"]["dvlbracket"] + \
        individual_dict_format["com_vec_dict"]["armbase"] + \
        individual_dict_format["com_vec_dict"]["shoulder"] + \
        individual_dict_format["com_vec_dict"]["upperarm"] + \
        individual_dict_format["com_vec_dict"]["elbow"] + \
        individual_dict_format["com_vec_dict"]["wrist"] + \
        individual_dict_format["com_vec_dict"]["jaw"] + \
        individual_dict_format["com_vec_dict"]["jaw_wrt_wrist"] + \
        [individual_dict_format["grav_mag_dict"]["vehicle"]] + \
        [individual_dict_format["grav_mag_dict"]["weightCA"]] + \
        [individual_dict_format["grav_mag_dict"]["weightBL"]] + \
        [individual_dict_format["grav_mag_dict"]["weightBR"]] + \
        [individual_dict_format["grav_mag_dict"]["dvl"]] + \
        [individual_dict_format["grav_mag_dict"]["dvlbracket"]] + \
        [individual_dict_format["grav_mag_dict"]["armbase"]] + \
        [individual_dict_format["grav_mag_dict"]["shoulder"]] + \
        [individual_dict_format["grav_mag_dict"]["upperarm"]] + \
        [individual_dict_format["grav_mag_dict"]["elbow"]] + \
        [individual_dict_format["grav_mag_dict"]["jaw"]] + \
        [individual_dict_format["grav_mag_dict"]["wrist"]] + \
        [individual_dict_format["drag"]["d_lin_angular"]] + \
        [individual_dict_format["drag"]["d_nonlin_angular"]] + \
        individual_dict_format["drag"]["d_lin_coeffs"] + \
        individual_dict_format["drag"]["d_nonlin_coeffs"] + \
        [individual_dict_format["link_volumes"]["shoulder"]] + \
        [individual_dict_format["link_volumes"]["upperarm"]] + \
        [individual_dict_format["link_volumes"]["elbow"]] + \
        [individual_dict_format["link_volumes"]["wrist"]] + \
        [individual_dict_format["link_volumes"]["armbase"]] + \
        [individual_dict_format["link_volumes"]["jaw"]] + \
        [individual_dict_format["link_masses"]["shoulder"]] + \
        [individual_dict_format["link_masses"]["upperarm"]] + \
        [individual_dict_format["link_masses"]["elbow"]] + \
        [individual_dict_format["link_masses"]["wrist"]] + \
        [individual_dict_format["link_masses"]["armbase"]] + \
        [individual_dict_format["link_masses"]["jaw"]] + \
        individual_dict_format["link_drags"]["shoulder"] + \
        individual_dict_format["link_drags"]["upperarm"] + \
        individual_dict_format["link_drags"]["elbow"] + \
        individual_dict_format["link_drags"]["wrist"] + \
        individual_dict_format["link_drags"]["jaw"]
    return individual_list_format

def calculateMSE(a, b):
    return (np.square(a-b)).mean()

def setupJulia():
    # ----------------------------------------------------------
    #                     Import Libraries
    # ----------------------------------------------------------

    # Main.eval('using Pkg')
    # Main.eval('Pkg.activate(".")')
    Main.eval('using RigidBodyDynamics, Rotations ')
    Main.eval('using LinearAlgebra, StaticArrays, DataStructures ')
    Main.eval('using MeshCat, MeshCatMechanisms, MechanismGeometries ')
    Main.eval('using CoordinateTransformations ')
    Main.eval('using GeometryBasics ')
    Main.eval('using Printf')
    # This Pkg.add line should not be necessary, but without it we get 
    # julia.core.JuliaError: Exception 'LoadError: InitError: UndefVarError: GR_jll not defined
    # Main.eval('import Pkg; Pkg.add("GR")')
    # Main.eval('using GR')
    Main.eval('using Plots')
    Main.eval('using CSV')
    Main.eval('using Tables')
    Main.eval('using ProgressBars')
    Main.eval('using Revise')
    Main.eval('using Random ')

    Main.eval('using DataFrames, StatsPlots, Interpolations ')

    Main.eval('include("HydroCalc.jl")')
    Main.eval('include("SimWExt.jl")')
    Main.eval('include("PIDCtlr.jl")')
    Main.eval('include("UVMSPlotting.jl")')
    Main.eval('include("HelperFuncs.jl")')
    Main.eval('include("Noiser.jl")')

    Main.eval('include("UVMSsetup.jl")')
    Main.eval('include("ConfigFiles/MagicNumPitchVal.jl")')
    Main.eval('include("ConfigFiles/ConstMagicNums.jl")')
    Main.eval('include("ConfigFiles/MagicNumBlueROVHardware.jl")')
    Main.eval('include("ConfigFiles/MagicNumAlpha.jl")')

    Main.eval('trajparsingfile = joinpath("..", "hinsdale_post_processing", "gettrajparamsfromyaml.jl") ')
    Main.eval('interpolationfile = joinpath("..", "hinsdale_post_processing", "mocap_interpolation.jl") ')
    Main.eval('simhelperfuncsfile = joinpath("..", "hinsdale_post_processing", "simcomparisonfuncs.jl") ')
    Main.eval('include(trajparsingfile) ')
    Main.eval('include(interpolationfile) ')

    Main.eval('urdf_file = joinpath("..","..","urdf", "blue_rov_hardware_fixedjaw_pythonhack.urdf") ')

    # ----------------------------------------------------------
    #                 One-Time Mechanism Setup
    # ----------------------------------------------------------

    Main.eval("""mech_blue_alpha, mvis, joint_dict, body_dict = mechanism_reference_setup(urdf_file)""")
    Main.eval("""include("TrajGenJoints.jl")""")
    Main.eval("include(simhelperfuncsfile)")

    Main.eval("""cob_frame_dict, com_frame_dict = setup_frames(body_dict, body_names, cob_vec_dict, com_vec_dict)""")
    Main.eval("""buoyancy_force_dict, gravity_force_dict = setup_buoyancy_and_gravity(buoyancy_mag_dict, grav_mag_dict)""")

    Main.eval("""state = MechanismState(mech_blue_alpha)""")
    Main.eval("""num_dofs = num_velocities(mech_blue_alpha)""")
    Main.eval("""num_actuated_dofs = num_dofs-2""")

    Main.eval('mech_blue_alpha, mvis, joint_dict, body_dict = mechanism_reference_setup(urdf_file)')
    Main.eval('include("TrajGenJoints.jl")')
    Main.eval('include(simhelperfuncsfile)')

    # Moving the includes for files with simulation tools here
    Main.eval('include("HydroCalc.jl")')
    Main.eval('include("PIDCtlr.jl")')

    return None

# Trying to set this up so that it can do everything that MainPitchValidation.jl does but in a single python script
def evalConfig(individual, trajectory_names):
    """This function produces an evaluation of the specific individual based on the given trajectories

    The individual is a dictionary full of configuration parameters for the simulator dynamics. The
    trajectories are specified by name. Valid names are hard-coded according to what was collected
    at Hinsdale. The output is a tuple of overall mean-square error
    
    Given an individual (configuration dictionary with specific dynamic parameters), and a set of trajectories to run on, this
    function computes the simulated alpha arm joint positions, and blue rov orientation.

    Then compares the simulated data to the measured arm joint positions, and imu orientation measurements
    """

    # ----------------------------------------------------------
    #  Redefine configuration parameters based on the EA parameters
    # ----------------------------------------------------------

    # First turn the list of parameters in the individual into an interpretable dictionary
    individual = individualListToDict(individual)

    # All the parameters that are going to be taken into functions
    Main.input_cob_vec_dict = individual["cob_vec_dict"]
    Main.eval('cob_vec_dict["vehicle"] = SVector{3, Float64}(input_cob_vec_dict["vehicle"])')
    Main.eval('cob_vec_dict["foamL"] = SVector{3, Float64}(input_cob_vec_dict["foamL"])')
    Main.eval('cob_vec_dict["foamR"] = SVector{3, Float64}(input_cob_vec_dict["foamR"])')
    Main.eval('cob_vec_dict["shoulder"] = SVector{3, Float64}(input_cob_vec_dict["shoulder"])')
    Main.eval('cob_vec_dict["upperarm"] = SVector{3, Float64}(input_cob_vec_dict["upperarm"])')
    Main.eval('cob_vec_dict["elbow"] = SVector{3, Float64}(input_cob_vec_dict["elbow"])')
    Main.eval('cob_vec_dict["wrist"] = SVector{3, Float64}(input_cob_vec_dict["wrist"])')
    Main.eval('cob_vec_dict["jaw"]  = SVector{3, Float64}(input_cob_vec_dict["jaw"])')

    Main.input_buoyancy_mag_dict = individual["buoyancy_mag_dict"]
    Main.eval('buoyancy_mag_dict["vehicle"] = input_buoyancy_mag_dict["vehicle"]')
    Main.eval('buoyancy_mag_dict["foamL"] = input_buoyancy_mag_dict["foamL"]')
    Main.eval('buoyancy_mag_dict["foamR"] = input_buoyancy_mag_dict["foamR"]')
    Main.eval('buoyancy_mag_dict["shoulder"] = input_buoyancy_mag_dict["shoulder"]')
    Main.eval('buoyancy_mag_dict["upperarm"] = input_buoyancy_mag_dict["upperarm"]')
    Main.eval('buoyancy_mag_dict["armbase"] = input_buoyancy_mag_dict["armbase"]')
    Main.eval('buoyancy_mag_dict["jaw"] = input_buoyancy_mag_dict["jaw"]')
    Main.eval('buoyancy_mag_dict["wrist"] = input_buoyancy_mag_dict["wrist"]')
    Main.eval('buoyancy_mag_dict["elbow"] = input_buoyancy_mag_dict["elbow"]')

    Main.input_com_vec_dict = individual["com_vec_dict"]
    Main.eval('com_vec_dict["vehicle"] = SVector{3, Float64}(input_com_vec_dict["vehicle"])')
    Main.eval('com_vec_dict["weightCA"] = SVector{3, Float64}(input_com_vec_dict["weightCA"])')
    Main.eval('com_vec_dict["weightBL"] = SVector{3, Float64}(input_com_vec_dict["weightBL"])')
    Main.eval('com_vec_dict["weightBR"] = SVector{3, Float64}(input_com_vec_dict["weightBR"])')
    Main.eval('com_vec_dict["dvl"] = SVector{3, Float64}(input_com_vec_dict["dvl"])')
    Main.eval('com_vec_dict["dvlbracket"] = SVector{3, Float64}(input_com_vec_dict["dvlbracket"])')
    Main.eval('com_vec_dict["armbase"] = SVector{3, Float64}(input_com_vec_dict["armbase"])')
    Main.eval('com_vec_dict["shoulder"] = SVector{3, Float64}(input_com_vec_dict["shoulder"])')
    Main.eval('com_vec_dict["upperarm"] = SVector{3, Float64}(input_com_vec_dict["upperarm"])')
    Main.eval('com_vec_dict["elbow"] = SVector{3, Float64}(input_com_vec_dict["elbow"])')
    Main.eval('com_vec_dict["wrist"] = SVector{3, Float64}(input_com_vec_dict["wrist"])')
    Main.eval('com_vec_dict["jaw"] = SVector{3, Float64}(input_com_vec_dict["jaw"])')
    Main.eval('com_vec_dict["jaw_wrt_wrist"] = SVector{3, Float64}(input_com_vec_dict["jaw_wrt_wrist"])')

    Main.input_grav_mag_dict = individual["grav_mag_dict"]
    Main.eval('grav_mag_dict["vehicle"] = input_grav_mag_dict["vehicle"]')
    Main.eval('grav_mag_dict["weightCA"] = input_grav_mag_dict["weightCA"]')
    Main.eval('grav_mag_dict["weightBL"] = input_grav_mag_dict["weightBL"]')
    Main.eval('grav_mag_dict["weightBR"] = input_grav_mag_dict["weightBR"]')
    Main.eval('grav_mag_dict["dvl"] = input_grav_mag_dict["dvl"]')
    Main.eval('grav_mag_dict["dvlbracket"] = input_grav_mag_dict["dvlbracket"]')
    Main.eval('grav_mag_dict["armbase"] = input_grav_mag_dict["armbase"]')
    Main.eval('grav_mag_dict["shoulder"] = input_grav_mag_dict["shoulder"]')
    Main.eval('grav_mag_dict["upperarm"] = input_grav_mag_dict["upperarm"]')
    Main.eval('grav_mag_dict["elbow"] = input_grav_mag_dict["elbow"]')
    Main.eval('grav_mag_dict["jaw"] = input_grav_mag_dict["jaw"]')
    Main.eval('grav_mag_dict["wrist"] = input_grav_mag_dict["wrist"]')

    Main.input_drag = individual["drag"]
    Main.eval('d_lin_angular = input_drag["d_lin_angular"]')
    Main.eval('d_nonlin_angular = input_drag["d_nonlin_angular"]')
    Main.eval('d_lin_coeffs = [input_drag["d_lin_coeffs"][1], input_drag["d_lin_coeffs"][2], input_drag["d_lin_coeffs"][3], d_lin_angular, d_lin_angular, d_lin_angular]')
    Main.eval('d_nonlin_coeffs = [input_drag["d_nonlin_coeffs"][1], input_drag["d_nonlin_coeffs"][2], input_drag["d_nonlin_coeffs"][3], d_nonlin_angular, d_nonlin_angular, d_nonlin_angular]')

    Main.input_link_volumes = individual["link_volumes"]
    Main.eval('link_volumes["shoulder"] = input_link_volumes["shoulder"]')
    Main.eval('link_volumes["upperarm"] = input_link_volumes["upperarm"]')
    Main.eval('link_volumes["elbow"] = input_link_volumes["elbow"]')
    Main.eval('link_volumes["wrist"] = input_link_volumes["wrist"]')
    Main.eval('link_volumes["armbase"] = input_link_volumes["armbase"]')
    Main.eval('link_volumes["jaw"] = input_link_volumes["jaw"]')

    Main.input_link_masses = individual["link_masses"]
    Main.eval('link_masses["shoulder"] = input_link_masses["shoulder"]')
    Main.eval('link_masses["upperarm"] = input_link_masses["upperarm"]')
    Main.eval('link_masses["elbow"] = input_link_masses["elbow"]')
    Main.eval('link_masses["wrist"] = input_link_masses["wrist"]')
    Main.eval('link_masses["armbase"] = input_link_masses["armbase"]')
    Main.eval('link_masses["jaw"] = input_link_masses["jaw"]')

    Main.input_link_drags = individual["link_drags"]
    Main.eval('link_drags["shoulder"] = SVector{3, Float64}(input_link_drags["shoulder"])')
    Main.eval('link_drags["upperarm"] = SVector{3, Float64}(input_link_drags["upperarm"])')
    Main.eval('link_drags["elbow"] = SVector{3, Float64}(input_link_drags["elbow"])')
    Main.eval('link_drags["wrist"] = SVector{3, Float64}(input_link_drags["wrist"])')
    Main.eval('link_drags["jaw"] = SVector{3, Float64}(input_link_drags["jaw"])')

    # Global parameters
    Main.d_lin_coeffs = individual["drag"]["d_lin_coeffs"] + [individual["drag"]["d_lin_angular"]]*3
    Main.d_nonlin_coeffs = individual["drag"]["d_nonlin_coeffs"] + [individual["drag"]["d_nonlin_angular"]]*3

    # Set the trajectories that we're going to be using for evaluation
    Main.all_traj_codes = trajectory_names

    # Defines cob_frame_dict, com_frame_dict, buoyancy_force_dict, gravity_force_dict
    # I can print these out and manually inspect them
    Main.eval('cob_frame_dict, com_frame_dict = setup_frames(body_dict, body_names, cob_vec_dict, com_vec_dict)')
    Main.eval('buoyancy_force_dict, gravity_force_dict = setup_buoyancy_and_gravity(buoyancy_mag_dict, grav_mag_dict)')

    # Defines state, num_dofs, num_actuated_dofs. I would think this is the same for each run..
    # Irrespective of the individual parameters
    Main.eval('state = MechanismState(mech_blue_alpha)')
    Main.eval('num_dofs = num_velocities(mech_blue_alpha)')
    Main.eval('num_actuated_dofs = num_dofs-2')

    # all_traj_codes is where I can add more of the trajectories to use for learning

    # ----------------------------------------------------------
    #                 Get Data for Comparison
    # ----------------------------------------------------------

    # Executing for loop (higher level) in python
    # Executing each portion of the for loop (lower level) in julia
    # I think this portion is what actually runs the simulator for each 
    # Of the specified trajectories
    all_imu_df = []
    all_js_df = []
    all_sim_df = []
    for (Main.i, Main.trial_code) in enumerate(Main.all_traj_codes):
        # I should have one entry (one in imu, one in joint) for each trajectory
        # This just prints the trial code. Should not leak or change
        Main.eval('println("This trial code: $(trial_code)")')
        # Defines params, des_df, sim_offset
        # Is sim_offset the settling time? This leads me to believe this should be consistent each run
        Main.eval('params, des_df, sim_offset = gettrajparamsfromyaml(trial_code, "otherhome", true)')
        # Get mocap data
        Main.eval('mocap_df = get_vehicle_response_from_csv(trial_code, "hinsdale-data-2023", false, true)')
        Main.eval('imu_df = get_imu_data_from_csv(trial_code, "hinsdale-data-2023", true)')
        Main.eval('imu_df = calc_rpy(imu_df)')
        Main.eval('js_df = get_js_data_from_csv(trial_code, "hinsdale-data-2023", "fullrange2", true)')
        # This is essentially grabbing data from files. Which should be the same every time this runs
        # so there should not be any stochasticity coming from here
        Main.eval('init_vs, init_vehpose = get_initial_vehicle_velocities(0, mocap_df)')
        Main.eval('init_quat, init_ωs = get_initial_conditions(0, imu_df)')

        # ----------------------------------------------------------
        #                         Simulate
        # ----------------------------------------------------------

        # Give the vehicle initial conditions from the mocap
        Main.eval('zero!(state)')
        Main.eval('set_configuration!(state, joint_dict["vehicle"], [init_quat..., init_vehpose...])')
        Main.eval('set_configuration!(state, joint_dict["base"], js_df[1,:axis_e_pos]-3.07)')
        Main.eval('set_configuration!(state, joint_dict["shoulder"], js_df[1,:axis_d_pos])')
        Main.eval('set_configuration!(state, joint_dict["elbow"], js_df[1,:axis_c_pos])')
        Main.eval('set_configuration!(state, joint_dict["wrist"], js_df[1,:axis_b_pos]-2.879)')
        Main.eval('init_vs_vector = FreeVector3D(root_frame(mech_blue_alpha), init_vs)')
        Main.eval('body_frame_init_vs = RigidBodyDynamics.transform(state, init_vs_vector, default_frame(body_dict["vehicle"]))')
        Main.eval('set_velocity!(state, joint_dict["vehicle"], [0, 0, 0, body_frame_init_vs.v...])')

        # Start up the controller
        # What is the noise cache?? That seems like something that would inject noise in the simulated trajectory
        Main.eval('noise_cache = NoiseCache(state)')
        Main.eval('filter_cache = FilterCache(state)')
        Main.eval('ctlr_cache = CtlrCache(state, noise_cache, filter_cache)')

        Main.eval('start_buffer = sim_offset+10')
        Main.eval('end_buffer = 5')
        Main.eval('delayed_params = delayedQuinticTrajParams(params,start_buffer, params.T+start_buffer)')

        # Simulate the trajectory
        # Stop this set of parameters early if it creates an exception in the math for the sim
        try:
            Main.eval("global ts, qs, vs = simulate_with_ext_forces(state, params.T+start_buffer+end_buffer, delayed_params, ctlr_cache, hydro_calc!, pid_control!; Δt=Δt)")
        except:
            error_fit = 1000*len(Main.all_traj_codes)
            return (error_fit, error_fit, error_fit, error_fit, error_fit, error_fit)
        # Main.eval("@show vs[end]'")

        # # ----------------------------------------------------------
        # # Prepare Plots (just what is necessary for csv generation)
        # # ----------------------------------------------------------

        # Main.eval('include("UVMSPlotting.jl")')
        # Main.eval('gr(size=(800, 800)) ')
        # Main.eval('@show sim_offset')

        # Main.eval('sim_palette = palette([:deepskyblue2, :magenta], 4)')
        # Main.eval('actual_palette = palette([:goldenrod1, :springgreen3], 4)')


        # Downsample the time steps to goal_freq
        Main.eval('ts_down = [ts[i] for i in 1:sample_rate:length(ts)]')
        Main.eval('ts_down_no_zero = ts_down[2:end]')

        # Set up data collection dicts
        Main.eval('paths = prep_actual_vels_and_qs_for_plotting(ts_down_no_zero)')
        Main.eval('sim_df = DataFrame(paths)')
        Main.eval('sim_df[!,"time_secs"] = ts_down_no_zero')

        Main.eval('deleteat!(sim_df, findall(<(10), sim_df[!,:time_secs]))')
        Main.eval('sim_df[!,:time_secs] = sim_df[!,:time_secs] .- minimum(sim_df[!,:time_secs])')

        # ----------------------------------------------------------
        #                 Save Trajectory to CSV
        # ----------------------------------------------------------

        # only save the trajectory if Joint 1 doesn't exceed the joint velocity limits (which is a proxy for indicating whether it is unstable)
        # Rows:
        # 1-10: Actual position data (qs)
        # 11-20: Actual velocity data (vs)
        # 21-30: Noisy position data (noisy_qs)
        # 31-40: Noisy velocity data (noisy_vs)
        # 41-44: Desired velocities 

        Main.eval('deleteat!(sim_df, 1:2:length(sim_df[!,:time_secs]))')
        Main.eval('const_dt_imu_df = interp_at_timesteps(sim_df[!,:time_secs], imu_df, [:roll, :pitch])')
        Main.eval('const_dt_js_df = interp_at_timesteps(sim_df[!,:time_secs], js_df, names(js_df))')

        all_imu_df.append(Main.const_dt_imu_df)
        all_js_df.append(Main.const_dt_js_df)
        all_sim_df.append(Main.sim_df)

    """For sim data, we've got
    q1 roll
    q2 pitch
    q3 yaw
    q4 vehicle x
    q5 vehicle y
    q6 vehicle z
    q7 joint E
    q8 joint D
    q9 joint C
    q10 joint B 
    """

    # Note: The imu_df and js_df are interpolated so they have a time column
    # The simulated data is not interpolated (I think) so they have no additional time column
    Main.current_imu_df = all_imu_df[0]
    Main.current_js_df = all_js_df[0]
    Main.current_sim_df = all_sim_df[0]

    # Get all the rows of the first column (timestamps)
    # Get all the rows of the second column (roll)
    # Get all the rows of the third column (pitch)
    # imu_ts = np.array(Main.eval("current_imu_df[:,1]"))
    imu_roll = np.array(Main.eval("current_imu_df[:,2]"))
    imu_pitch = np.array(Main.eval("current_imu_df[:,3]"))

    # Get all of the joint state information from the actual trajectory
    meas_joint_e = np.array(Main.eval("current_js_df[:,6]"))
    meas_joint_d = np.array(Main.eval("current_js_df[:,5]"))
    meas_joint_c = np.array(Main.eval("current_js_df[:,4]"))
    meas_joint_b = np.array(Main.eval("current_js_df[:,3]"))

    # Get all of the simulated information
    sim_roll = np.array(Main.eval("current_sim_df[:,1]"))
    sim_pitch = np.array(Main.eval("current_sim_df[:,2]"))
    sim_joint_e = np.array(Main.eval("current_sim_df[:,7]"))
    sim_joint_d = np.array(Main.eval("current_sim_df[:,8]"))
    sim_joint_c = np.array(Main.eval("current_sim_df[:,9]"))
    sim_joint_b = np.array(Main.eval("current_sim_df[:,10]"))

    # Calculate mean square error for all of our metrics
    mse_roll = calculateMSE(sim_roll, imu_roll)
    mse_pitch = calculateMSE(sim_pitch, imu_pitch)
    mse_joint_e = calculateMSE(sim_joint_e, meas_joint_e)
    mse_joint_d = calculateMSE(sim_joint_d, meas_joint_d)
    mse_joint_c = calculateMSE(sim_joint_c, meas_joint_c)
    mse_joint_b = calculateMSE(sim_joint_b, meas_joint_b)

    return (mse_roll, mse_pitch, mse_joint_e, mse_joint_d, mse_joint_c, mse_joint_b)

def evalConfigSingleObj(individual, trajectory_names):
    mse_roll, mse_pitch, mse_joint_e, mse_joint_d, mse_joint_c, mse_joint_b = evalConfig(individual, trajectory_names)
    return np.sum([mse_roll, mse_pitch, mse_joint_e, mse_joint_d, mse_joint_c, mse_joint_b]),

# This needs to be called before running evalConfig(). This is the one time setup of all the necessary libraries
# and parameters for running the eval code multiple times
setupJulia()

default_params_dict = {
    "cob_vec_dict": {
        "vehicle": [0.0074, 0.0, 0.02],
        "foamL": [0.0, 0.11, 0.027], # guess
        "foamR": [0.0, -0.11, 0.027], #guess
        "shoulder": [-0.001, -0.003, 0.032],
        "upperarm": [0.073, 0.0, -0.002],
        "elbow": [0.003, 0.001, -0.017],
        "wrist": [0.0, 0.0, -0.098],
        "jaw": [0.0, 0.0, 0.0]
    },
    "buoyancy_mag_dict": {
        "shoulder": 0.17605026,
        "upperarm": 1.9854557100000003,
        "foamR": 8.86,
        "armbase": 1.97567514,
        "jaw": 0.19561140000000005,
        "wrist": 1.5159883500000002,
        "vehicle": 128.33442000000002, #volume times gravity
        "foamL": 8.46,
        "elbow": 0.24451425000000004
    },
    "com_vec_dict": {
        "armbase": [-0.075, -0.006, -0.003],
        "vehicle": [0.0, 0.0, 0.0],
        "shoulder": [0.005, -0.001, 0.016],
        "upperarm": [0.073, 0.0, 0.0],
        "wrist": [0.0, 0.0, -0.098],
        "elbow": [0.017, -0.026, -0.002],
        "jaw": [0.0, 0.0, 0.0],
        "jaw_wrt_wrist": [0.0, 0.0, -0.19],
        "weightCA": [-0.2, 0.165, -0.075], # guess
        "weightBL": [-0.0975, 0.1275, -0.1325], # guess
        "weightBR": [-0.0975, -0.1275, -0.1325], # guess
        "dvlbracket": [-0.1542, 0.0439, -0.0795], # guess
        "dvl": [-0.1887, 0.0439, -0.0595], # guess
    },
    "grav_mag_dict": {
        "armbase": 3.3452100000000002,
        "vehicle": 129.1977, # Weight * gravity
        "shoulder": 1.90314,
        "upperarm": 4.20849,
        "weightBR": 1.62,
        "dvl": 0.69,
        "elbow": 1.1281500000000002,
        "jaw": 0.49050000000000005,
        "weightCA": 3.24,
        "weightBL": 1.62,
        "wrist": 3.2667300000000004,
        "dvlbracket": 1.01
    },
    "drag": {
        "d_lin_angular": 0.07,
        "d_nonlin_angular": 1.55,
        "d_lin_coeffs": [4.03, 6.22, 5.18], # The next 3 coefficients are d_lin_angular
        "d_nonlin_coeffs": [18.18, 21.66, 36.99] # The next 3 coefficients are d_nonlin_angular
    },
    "link_volumes": {
        "shoulder": 0.018,
        "upperarm": 0.203,
        "elbow": 0.025,
        "wrist": 0.155,
        "armbase": 0.202,
        "jaw": 0.02
    },
    "link_masses": {
        "shoulder": 0.194,
        "upperarm": 0.429,
        "elbow": 0.115,
        "wrist": 0.333,
        "armbase": 0.341,
        "jaw": 0.05
    },
    "link_drags": {
        "shoulder": [0.26, 0.26, 0.3],
        "upperarm": [0.3, 1.6, 1.6],
        "elbow": [0.26, 0.3, 0.26],
        "wrist": [1.8, 1.8, 0.3],
        "jaw": [.05, .05, .05]
    }
}
default_params_individual = individualDictToList(default_params_dict)
import os, psutil
process = psutil.Process()

import random
from deap import creator, base, tools, algorithms

# https://deap.readthedocs.io/en/master/tutorials/basic/part1.html
# This is a link to the docs I'm using

# Setup the directory for saving data
save_dir = os.path.expanduser("~")+"/hpc-share/uvms/preliminary/trial_0"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
# Create csv files for saving data
with open(save_dir+"/fitness.csv", 'w') as file:
    file.write("generation, top_fitness")
    file.write('\n')
with open(save_dir+"/best_solution.csv", 'w') as file:
    file.write("generation, best_solution")
    file.write('\n')

solution_save_interval = 50 # generations

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=119)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalConfigSingleObj, trajectory_names=["_alt_001-0"])

# Initialize a population
population_size = 2
population = toolbox.population(n=1)
# Replace the first individual's parameters with the default parameters
population[0][:] = default_params_individual[:]
# Create a n-1 size population that has mutations of the default parameters
mutated_default_param_inds = algorithms.varAnd(population*(population_size-1), toolbox, cxpb=0.5, mutpb=0.1)
# Put both populations together
population = population + mutated_default_param_inds

fitnesses = []
best_individuals = []

# Evaluate the starting population first at generation 0
fits = toolbox.map(toolbox.evaluate, population)

# Assign fitnesses
for fit, ind in zip(fits, population):
    ind.fitness.values = fit

# Rename to keep naming consistent
offspring = population

# Then perform the mutations
population = toolbox.select(offspring, k=len(population))


# Append the best solution and its fitness after evaluation but before selection
top1 = tools.selBest(offspring, k=1)[0]
# This is just getting the best one from the population
# and then taking it out of the list form so
# we can print it nicely to see what the solution is
fitnesses.append(top1.fitness.values[0])
best_individuals.append(top1)

# Save the best solution and the fitness of generation 0
with open(save_dir+"/fitness.csv", 'a') as file:
    fitness_str = ','.join([str(0), str(top1.fitness.values[0])])
    file.write(fitness_str+'\n')

with open(save_dir+"/best_solution.csv", 'a') as file:
    individual_str = ','.join([str(x) for x in top1])
    file.write(str(0)+","+individual_str+'\n')

# Number of generations after generation 0 (which is initial population)
NGEN=3
for gen in range(NGEN):
    # This is the typical generations loop
    # Run this for a set number of generations

    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    # not sure what varAnd does. Also not sure what the algorithms library is. Looks like it's something within DEAP
    # offspring... hm.. not sure how that works in this context

    fits = toolbox.map(toolbox.evaluate, offspring)
    # mapping the evaluate function to each of the offspring?
    # What is the difference between the offspring and the population?
    # If it is the offspring after selection, then I would expect to see some selection
    # operation performed on the population and offspring being the output of that
    # ah, fits = fitnesses

    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
        # So I guess this is how we assign a fitness to each offspring?
        # Before we only calculated the fitnesses but didn't do anything with them
        # We just had them in a list


    # Append the best solution and its fitness after evaluation but before selection
    top1 = tools.selBest(offspring, k=1)[0]
    # This is just getting the best one from the population
    # and then taking it out of the list form so
    # we can print it nicely to see what the solution is
    fitnesses.append(top1.fitness.values[0])
    best_individuals.append(top1)

    # Save the best solution and the fitness at regular intervals
    with open(save_dir+"/fitness.csv", 'a') as file:
        fitness_str = ','.join([str(gen+1), str(top1.fitness.values[0])])
        file.write(fitness_str+'\n')

    if (gen+1) % solution_save_interval == 0:
        with open(save_dir+"/best_solution.csv", 'a') as file:
            individual_str = ','.join([str(x) for x in top1])
            file.write(str(gen+1)+","+individual_str+'\n')
    
    population = toolbox.select(offspring, k=len(population))
    # Then we select the offspring, presumably based on fitness, and we select
    # the amount equal to the amount we need in the population

