include("MagicNumBlueROVHardware.jl")

mutable struct MagicNumBlueRovWithAlphaArm
    blue_rov::MagicNumBlueRov
    link_drags::Dict{String, SVector{3, Float64}}
    joint_lim_dict::Dict{String, SVector{2, Float64}}
    vel_lim_dict::Dict{String, SVector{2, Float64}}
end

function create_magic_num_bluerov_with_alpha_arm(rho::Float64)::MagicNumBlueRovWithAlphaArm
    # Include MagicNumBlueROV before this, to add on to the dictionary created there
    blue_rov = create_magic_num_bluerov()
    # Center of Buoyancy vectors for the arm links
    blue_rov.cob_vec_dict["shoulder"] = SVector{3, Float64}([-.001, -.003, .032])
    blue_rov.cob_vec_dict["upperarm"] = SVector{3, Float64}([.073, 0.0, -.002])
    blue_rov.cob_vec_dict["elbow"] = SVector{3, Float64}([.003, .001, -.017])
    blue_rov.cob_vec_dict["wrist"] = SVector{3, Float64}([0.0, 0.0, -0.098])
    blue_rov.cob_vec_dict["jaw"] = SVector{3, Float64}([0.0, 0.0, 0.0])
    # TODO double check these numbers against the documentation
    blue_rov.com_vec_dict["armbase"] = SVector{3, Float64}([-.075, -.006, -.003])

    # Center of mass vectors for the arm links
    blue_rov.com_vec_dict["shoulder"] = SVector{3, Float64}([0.005, -.001, 0.016])
    blue_rov.com_vec_dict["upperarm"] = SVector{3, Float64}([.073, 0.0, 0.0])
    blue_rov.com_vec_dict["elbow"] = SVector{3, Float64}([.017, -.026, -.002])
    blue_rov.com_vec_dict["wrist"] = SVector{3, Float64}([0.0, 0.0, -.098])
    blue_rov.com_vec_dict["jaw"] = SVector{3, Float64}([0.0, 0.0, 0.0])
    blue_rov.com_vec_dict["armbase"] = SVector{3, Float64}([-.075, -.006, -.003])
    blue_rov.com_vec_dict["jaw_wrt_wrist"] = SVector{3, Float64}([0., 0., -.190])

    # Calculate buoyancy forces
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
        blue_rov.buoyancy_mag_dict[k] = v*rho*9.81*.001
    end

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
        blue_rov.grav_mag_dict[k] = v*9.81
    end

    # Drag forces on arm
    # link_drags = Dict("shoulder" => [0.26 0.26 0.3]*rho,
    #                     "upperarm" => [0.3 1.6 1.6]*rho,
    #                     "elbow" => [0.26 0.3 0.26]*rho,
    #                     "wrist" => [1.8 1.8 0.3]*rho,
    #                     "jaw" => [.05, .05, .05]*rho)

    link_drags = Dict{String, SVector{3, Float64}}(
        "shoulder" => [0.26, 0.26, 0.3],
        "upperarm" => [0.3, 1.6, 1.6],
        "elbow" => [0.26, 0.3, 0.26],
        "wrist" => [1.8, 1.8, 0.3],
        "jaw" => [.05, .05, .05]
    )


    # Arm position joint limits
    joint_lim_dict = Dict{String, SVector{2, Float64}}(
        "base" => [-175*pi/180, 175*pi/180],
        "shoulder" => [0, 200*pi/180],
        "elbow" => [0, 200*pi/180],
        "wrist" => [-165*pi/180, 165*pi/180],
        "jaw" => [0., 0.022]
    )

    θb = deg2rad(30)
    θc = deg2rad(50)
    vel_lim_dict = Dict{String, SVector{2, Float64}}(
        "base" => [-θb, θb],
        "shoulder" => [-θb, θb],
        "elbow" => [-θb, θb],
        "wrist" => [-θc, θc],
        "jaw" => [-.003, .003]
    )
    MagicNumBlueRovWithAlphaArm(blue_rov, link_drags, joint_lim_dict, vel_lim_dict)
end
