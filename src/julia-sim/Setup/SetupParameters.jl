include("Parameters.jl")

function setup_parameters()::SimulationParameters
    # ----------------------------------------------------------
    # Parameters inherent to the simulated world
    # ----------------------------------------------------------
    rho = 997     # kg/m^3 (density of the water)
    Δt = 1.e-3    # simulation time step
    world_parameters = WorldParameters(
        rho,
        Δt
    )
    # ----------------------------------------------------------
    # Parameters inherent to the UVMS and its dynamics
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # Starting with limits for the actuator
    # ----------------------------------------------------------
    # Arm position joint limits
    joints = Dict{String, SVector{2, Float64}}(
        "base" => [-175*pi/180, 175*pi/180],
        "shoulder" => [0, 200*pi/180],
        "elbow" => [0, 200*pi/180],
        "wrist" => [-165*pi/180, 165*pi/180],
        "jaw" => [0., 0.022]
    )
    # Arm velocity limits
    θb = deg2rad(30)
    θc = deg2rad(50)
    velocities = Dict{String, SVector{2, Float64}}(
        "base" => [-θb, θb],
        "shoulder" => [-θb, θb],
        "elbow" => [-θb, θb],
        "wrist" => [-θc, θc],
        "jaw" => [-.003, .003]
    )
    # Actuator upper torque limits
    torques = Dict(
        "roll" => 0.0,
        "pitch" => 0.0,
        "yaw" => 20.,
        "x" => 71.5,
        "y" => 88.2,
        "z" => 177.,
        "base" => 10.,
        "shoulder" => 10.,
        "elbow" => 10.,
        "wrist" => 0.6,
        "jaw" => 0.6
    )
    # D_tau limits for a controller at 100Hz
    thruster_dtau_lim = 0.001
    joint_dtau_lim = 0.1
    d_taus = Dict(
        "roll" => 0.0,
        "pitch" => 0.0,
        "yaw" => thruster_dtau_lim,
        "x" => thruster_dtau_lim,
        "y" => thruster_dtau_lim,
        "z" => thruster_dtau_lim,
        "base" => joint_dtau_lim,
        "shoulder" => joint_dtau_lim,
        "elbow" => joint_dtau_lim,
        "wrist" => .005,
        "jaw" => joint_dtau_lim
    )
    actuator_limits = ActuatorLimits(
        joints,
        velocities,
        torques,
        d_taus
    )
    # ----------------------------------------------------------
    # Buoyancy parameters for both vehicle and arm
    # ----------------------------------------------------------
    # Center of Buoyancy vectors
    cob = Dict{String, SVector{3,Float64}}()
    cob["vehicle"] = SVector{3, Float64}([0.0074, 0.0, 0.02])
    cob["foamL"] = SVector{3, Float64}([0.00, .11, 0.027]) # guess
    cob["foamR"] = SVector{3, Float64}([0.00, -.11, 0.027]) #guess
    cob["shoulder"] = SVector{3, Float64}([-.001, -.003, .032])
    cob["upperarm"] = SVector{3, Float64}([.073, 0.0, -.002])
    cob["elbow"] = SVector{3, Float64}([.003, .001, -.017])
    cob["wrist"] = SVector{3, Float64}([0.0, 0.0, -0.098])
    cob["jaw"] = SVector{3, Float64}([0.0, 0.0, 0.0])

    # Calculate buoyancy magnitudes
    buoyancy_magnitudes = Dict{String, Float64}()
    buoyancy_magnitudes["vehicle"] = 13.082*9.81 #(volume * gravity)
    buoyancy_magnitudes["foamL"] = 8.2
    buoyancy_magnitudes["foamR"] = 9.12
    buoyancy_magnitudes["foamL"] = 8.46
    buoyancy_magnitudes["foamR"] = 8.66+.2
    link_volumes = Dict(
        "shoulder" =>   0.018, # volume in L
        "upperarm" =>   0.203,
        "elbow" =>      0.025,
        "wrist" =>      0.155,
        "armbase" =>    0.202,
        "jaw" =>        0.02
    ) # reasonable estimate
    # f = 997 (kg/m^3) * 9.81 (m/s^2) * V_in_L *.001 (m^3) = kg m / s^2
    for (k,v) in link_volumes
        buoyancy_magnitudes[k] = v*rho*9.81*.001
    end
    # This will remain empty until defined later on
    buoyancy_forces = Dict{String, FreeVector3D}()
    buoyancy_parameters = BuoyancyParameters(
        cob,
        buoyancy_magnitudes,
        buoyancy_forces
    )
    # ----------------------------------------------------------
    # Gravity parameters for both vehicle and arm
    # ----------------------------------------------------------
    # Center of mass vectors for vehicle
    com = Dict{String, SVector{3, Float64}}()
    com["vehicle"] = SVector{3, Float64}([0.0, 0.0, 0.0])
    com["weightCA"] = SVector{3, Float64}([-.20, .165, -.075]) # guess
    com["weightBL"] = SVector{3, Float64}([-.0975, .1275, -.1325]) # guess
    com["weightBR"] = SVector{3, Float64}([-.0975, -.1275, -.1325]) # guess
    com["dvl"] = SVector{3, Float64}([-.1887, .0439, -.1095+0.05]) # guess
    com["dvlbracket"] = SVector{3, Float64}([-.1887+.0345, .0439, -.1295+.05]) # guess
    # Center of mass vectors for the arm links
    com["shoulder"] = SVector{3, Float64}([0.005, -.001, 0.016])
    com["upperarm"] = SVector{3, Float64}([.073, 0.0, 0.0])
    com["elbow"] = SVector{3, Float64}([.017, -.026, -.002])
    com["wrist"] = SVector{3, Float64}([0.0, 0.0, -.098])
    com["jaw"] = SVector{3, Float64}([0.0, 0.0, 0.0])
    com["armbase"] = SVector{3, Float64}([-.075, -.006, -.003])
    com["jaw_wrt_wrist"] = SVector{3, Float64}([0., 0., -.190])

    # Magnitudes of gravitational forces
    gravity_magnitudes = Dict{String, Float64}()
    gravity_magnitudes["vehicle"] = 13.17*9.81 #(weight * gravity)
    gravity_magnitudes["weightCA"] = 3.24 # water weight
    gravity_magnitudes["weightBL"] = 1.62 # water weight
    gravity_magnitudes["weightBR"] = 1.62 # water weight
    gravity_magnitudes["dvl"] = 0.69 # water weight
    gravity_magnitudes["dvlbracket"] = 1.01 # water weight
    # Calculate gravitational forces
    link_masses = Dict(
        "shoulder" =>    0.194,
        "upperarm" =>   0.429,
        "elbow" =>      0.115,
        "wrist" =>      0.333,
        "armbase" =>    0.341,
        "jaw" =>        0.05
    )
    for (k,v) in link_masses
        gravity_magnitudes[k] = v*9.81
    end

    # This will be populated later
    gravity_forces = Dict{String, FreeVector3D}()

    gravity_parameters = GravityParameters(
        com,
        gravity_magnitudes,
        gravity_forces
    )
    # ----------------------------------------------------------
    # Drag parameters
    # ----------------------------------------------------------
    link_drags = Dict{String, SVector{3, Float64}}(
        "shoulder" => [0.26, 0.26, 0.3],
        "upperarm" => [0.3, 1.6, 1.6],
        "elbow" => [0.26, 0.3, 0.26],
        "wrist" => [1.8, 1.8, 0.3],
        "jaw" => [.05, .05, .05]
    )
    d_lin_angular = .07
    d_nonlin_angular = 1.55
    d_lin_coeffs = [4.03, 6.22, 5.18, d_lin_angular, d_lin_angular, d_lin_angular]
    d_nonlin_coeffs = [18.18, 21.66, 36.99, d_nonlin_angular, d_nonlin_angular, d_nonlin_angular]
    drag_parameters = DragParameters(
        link_drags,
        d_lin_coeffs,
        d_nonlin_coeffs
    )
    # ----------------------------------------------------------
    # Noise parameters
    # ----------------------------------------------------------
    # Sensor noise distributions
    # Encoder --> joint position noise -integration-> joint velocity noise
    # Gyroscope --> vehicle body vel noise
    v_ang_vel_noise_dist = Distributions.Normal(0, 0) # .0013) # 75 mdps (LSM6DSOX)
    arm_pos_noise_dist = Distributions.Normal(0, 0) # .0017/6) # .1 degrees, from Reach website
    accel_noise_dist = Distributions.Normal(0, 0) # 0.017658/10) # 1.8 mg = .0176 m/s2 (LSM6DSOX)
    gyro_rand_walk_dist = Distributions.Normal(0, 0) # .000001)
    accel_rand_walk_dist = Distributions.Normal(0, 0)#0.00001)
    noise_parameters = NoiseParameters(
        v_ang_vel_noise_dist,
        arm_pos_noise_dist,
        accel_noise_dist,
        gyro_rand_walk_dist,
        accel_rand_walk_dist
    )
    # ----------------------------------------------------------
    # Now we wrap up UVMS parameters into the UVMS model
    # ----------------------------------------------------------
    is_jaw_fixed = true
    body_names = ["vehicle", "shoulder", "upperarm", "elbow", "wrist"]
    dof_names = ["roll", "pitch", "yaw", "x", "y", "z", "base", "shoulder", "elbow", "wrist"]
    vehicle_extras = ["weightCA", "weightBL", "weightBR", "dvl", "dvlbracket", "foamL", "foamR"]
    model_parameters = ModelParameters(
        is_jaw_fixed,
        body_names,
        dof_names,
        actuator_limits,
        buoyancy_parameters,
        gravity_parameters,
        vehicle_extras,
        drag_parameters,
        noise_parameters
    )
    # ----------------------------------------------------------
    # Moving onto controller parameters
    # ----------------------------------------------------------
    add_noise = true
    Ku_wrist = 1.35e-3
    Tu_wrist = 0.16
    v_Kp = 1.2
    Kp = .7
    Kp_dict = Dict(
        "yaw" => 2.0,
        "x" => v_Kp,
        "y" => v_Kp,
        "z" => v_Kp,
        "base" => 3.38e-2,
        "shoulder" => 4.59e-2,
        "elbow" => 1.35e-2,
        "wrist" => Kp*Ku_wrist,
        "jaw" => 5.4e-4
    )
    v_Ki = 0.6
    Ki = 1.75
    Ki_dict = Dict(
        "yaw" => 1.,
        "x" => v_Ki,
        "y" => v_Ki,
        "z" => v_Ki,
        "base" => 4.39e-1,
        "shoulder" => 3.73e-1,
        "elbow" => 9.64e-2,
        "wrist" => Ki*Ku_wrist/Tu_wrist,
        "jaw" => 3.6e-3
    )
    v_Kd = 1.61
    Kd = 0.105
    Kd_dict = Dict(
        "yaw" => 0.1,
        "x" => v_Kd,
        "y" => v_Kd,
        "z" => v_Kd,
        "base" => 1.74e-3,
        "shoulder" => 3.82e-3,
        "elbow" => 1.27e-3,
        "wrist" => Kd*Ku_wrist*Tu_wrist,
        "jaw" => 2.03e-5
    )
    do_feedforward = true
    feedforward_propogation = 0.25
    filtering_kernel = 5
    ctrl_freq = Int(100)
    ctrl_steps = 4*(1/Δt)/ctrl_freq
    @assert rem(ctrl_steps, 1) == 0.0 "Please pick a control frequency that is an even divisor of the simulation frequency."
    controller_parameters = ControllerParameters(
        add_noise,
        Kp_dict,
        Ki_dict,
        Kd_dict,
        do_feedforward,
        feedforward_propogation,
        filtering_kernel,
        ctrl_freq,
        ctrl_steps
    )
    return SimulationParameters(
        world_parameters,
        model_parameters,
        controller_parameters
    )
end