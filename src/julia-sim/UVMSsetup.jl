using MeshCat
using Sockets
using CSV
using DataFrames

function setup_frames(body_dict, body_name_list, cob_vec_dict, com_vec_dict)
    cob_frame_dict = Dict{String, CartesianFrame3D}()
    com_frame_dict = Dict{String, CartesianFrame3D}()
    vis_element = 1

    for (i, body_name) in enumerate(body_name_list)
        # Get the body of interest
        bod = body_dict[body_name]

        # Name the two new frames
        frame_cob = CartesianFrame3D(body_name*"_cob")
        frame_com = CartesianFrame3D(body_name*"_com")

        # Define the two new frames wrt the default frame
        cob_transform = Transform3D(frame_cob, default_frame(bod), cob_vec_dict[body_name])
        com_transform = Transform3D(frame_com, default_frame(bod), com_vec_dict[body_name])

        # Add the frames to the bodies
        if !(RigidBodyDynamics.is_fixed_to_body(bod, frame_cob))
            add_frame!(bod, cob_transform)
            cob_frame_dict[body_name] = frame_cob
        end
        if !(RigidBodyDynamics.is_fixed_to_body(bod, frame_com))
            add_frame!(bod, com_transform)
            com_frame_dict[body_name] = frame_com
        end

        # if desired, visualize the new frames on the body
        if i == vis_element
            # setelement!(mvis, default_frame(bod))
            setelement!(mvis, frame_cob, 0.35)
            setelement!(mvis, frame_com, 0.25)
        end

    end

    # Add buoyancy or gravity to each extra element added to the BlueROV
    if @isdefined(vehicle_extras_list)
        vehicle_body = body_dict["vehicle"]

        for (j, item_name) in enumerate(vehicle_extras_list)
            # println("Parsing item "*item_name)
            frame_center = CartesianFrame3D(item_name*"_centerframe")
            if haskey(com_vec_dict, item_name)
                # println("Has mass")
                center_transform = Transform3D(frame_center, default_frame(vehicle_body), com_vec_dict[item_name])
                com_frame_dict[item_name] = frame_center
            elseif haskey(cob_vec_dict, item_name)
                # println("has buoyancy")
                center_transform = Transform3D(frame_center, default_frame(vehicle_body), cob_vec_dict[item_name])
                cob_frame_dict[item_name] = frame_center
            end

            if !(RigidBodyDynamics.is_fixed_to_body(vehicle_body, frame_center))
                add_frame!(vehicle_body, center_transform)
            end

            # if item_name == "dvl" || item_name == "dvlbracket"
            #     println("Trying to show com")
            #     setelement!(mvis, frame_center, 0.25)
            # end

        end
    end

    if haskey(body_dict, "jaw") == false
        jaw_wrt_wristframe = com_vec_dict["jaw_wrt_wrist"]
        jaw_com_frame = CartesianFrame3D("jaw_com_cob_wrt_wrist")
        com_transform = Transform3D(jaw_com_frame, default_frame(body_dict["wrist"]), jaw_wrt_wristframe)
        if !(RigidBodyDynamics.is_fixed_to_body(body_dict["wrist"], jaw_com_frame))
            add_frame!(body_dict["wrist"], com_transform)
            cob_frame_dict["jaw_wrt_wrist"] = jaw_com_frame
            com_frame_dict["jaw_wrt_wrist"] = jaw_com_frame
            setelement!(mvis, jaw_com_frame)
        end
    end

    alphabase_com_wrt_linkframe = com_vec_dict["armbase"]
    # Arm base is rigidly attached to vehicle, so it has a transform in the vehicle's frame. It's the 5th body in the URDF attached to the vehicle.
    linkframe_wrt_vehframe = translation(RigidBodyDynamics.frame_definitions(body_dict["vehicle"])[5])
    # # IF THE ARM IS ROTATED THIS HAS TO CHANGE!!!!
    alphabase_com_wrt_vehframe = alphabase_com_wrt_linkframe + linkframe_wrt_vehframe
    alphabase_com_frame = CartesianFrame3D("armbase_com_cob")
    com_transform = Transform3D(alphabase_com_frame, default_frame(body_dict["vehicle"]), alphabase_com_wrt_vehframe)

    if !(RigidBodyDynamics.is_fixed_to_body(body_dict["vehicle"], alphabase_com_frame))
        add_frame!(body_dict["vehicle"], com_transform)
        cob_frame_dict["armbase"] = alphabase_com_frame
        com_frame_dict["armbase"] = alphabase_com_frame
        # setelement!(mvis, alphabase_com_frame)
    end
    # print("THIS SHOULD SAY after_arm_to_vehicle: ")
    println(RigidBodyDynamics.frame_definitions(body_dict["vehicle"])[5].from)
    return cob_frame_dict, com_frame_dict
end

function mechanism_reference_setup(urdf_file, default_port=0)
    # vis = Visualizer(
    #     MeshCat.CoreVisualizer(
    #         default_port=default_port
    #     ),
    #     ["meshcat"]
    # )
    # vis = Visualizer()
    vis = Visualizer(
        MeshCat.CoreVisualizer(ip"127.0.0.1", default_port),
        ["meshcat"]
    )
    mech_blue_alpha = parse_urdf(urdf_file; floating=true, gravity = [0.0, 0.0, 0.0])
    # delete!(vis)

    # Create visuals of the URDFs
    mvis = MechanismVisualizer(mech_blue_alpha, URDFVisuals(urdf_file), vis[:alpha])

    # Name the joints and bodies of the mechanism
    joint_dict = Dict{String, RigidBodyDynamics.Joint}()
    body_dict = Dict{String, RigidBodyDynamics.RigidBody}()
    for (idx, link_name) in enumerate(body_names)
        body_dict[link_name] = bodies(mech_blue_alpha)[idx+1]
    end
    joint_dict["vehicle"] = joints(mech_blue_alpha)[1]
    for (idx, dof_name) in enumerate(dof_names)
        if idx > 6
            joint_dict[dof_name] = joints(mech_blue_alpha)[idx-5]
        end
    end
    return mech_blue_alpha, mvis, joint_dict, body_dict
end

function setup_buoyancy_and_gravity(buoyancy_mag_dict, grav_mag_dict)
    for (k, mag) in buoyancy_mag_dict
        buoyancy_force_dict[k] = FreeVector3D(root_frame(mech_blue_alpha), [0.0, 0.0, mag])
    end
    for (k, mag) in grav_mag_dict
        gravity_force_dict[k] = FreeVector3D(root_frame(mech_blue_alpha), [0.0, 0.0, -mag])
    end
    return buoyancy_force_dict, gravity_force_dict
end

function get_param(row, param)
    row[!, param][1]
end

function load_parameters_from_csv(csv_dir, target_index)
    global d_lin_angular, d_nonlin_angular, d_lin_coeffs, d_nonlin_coeffs
    global cob_frame_dict, com_frame_dict, buoyancy_force_dict, gravity_force_dict, state, num_dofs, num_actuated_dofs
    df = CSV.read(csv_dir, DataFrame)
    r = first(df[df."individual_index" .== target_index, :], 1)

    # cob_vec_dict already exists. Overwrite what is inside
    cob_vec_dict["vehicle"] = SVector{3, Float64}(
        get_param(r, "cob_vec_dict:vehicle:0"),
        get_param(r, "cob_vec_dict:vehicle:1"),
        get_param(r, "cob_vec_dict:vehicle:2")
    )
    cob_vec_dict["foamL"] = SVector{3, Float64}(
        get_param(r, "cob_vec_dict:foamL:0"),
        get_param(r, "cob_vec_dict:foamL:1"),
        get_param(r, "cob_vec_dict:foamL:2")
    )

    cob_vec_dict["foamR"] = SVector{3, Float64}(
        get_param(r, "cob_vec_dict:foamR:0"),
        get_param(r, "cob_vec_dict:foamR:1"),
        get_param(r, "cob_vec_dict:foamR:2")
    )

    cob_vec_dict["shoulder"] = SVector{3, Float64}(
        get_param(r, "cob_vec_dict:shoulder:0"),
        get_param(r, "cob_vec_dict:shoulder:1"),
        get_param(r, "cob_vec_dict:shoulder:2")
    )

    cob_vec_dict["upperarm"] = SVector{3, Float64}(
        get_param(r, "cob_vec_dict:upperarm:0"),
        get_param(r, "cob_vec_dict:upperarm:1"),
        get_param(r, "cob_vec_dict:upperarm:2")
    )

    cob_vec_dict["elbow"] = SVector{3, Float64}(
        get_param(r, "cob_vec_dict:elbow:0"),
        get_param(r, "cob_vec_dict:elbow:1"),
        get_param(r, "cob_vec_dict:elbow:2")
    )

    cob_vec_dict["wrist"] = SVector{3, Float64}(
        get_param(r, "cob_vec_dict:wrist:0"),
        get_param(r, "cob_vec_dict:wrist:1"),
        get_param(r, "cob_vec_dict:wrist:2")
    )

    cob_vec_dict["jaw"] = SVector{3, Float64}(
        get_param(r, "cob_vec_dict:jaw:0"),
        get_param(r, "cob_vec_dict:jaw:1"),
        get_param(r, "cob_vec_dict:jaw:2")
    )

    # Overwrite buoyancy_mag_dict
    buoyancy_mag_dict["vehicle"] = get_param(r, "buoyancy_mag_dict:vehicle")
    buoyancy_mag_dict["foamL"] = get_param(r, "buoyancy_mag_dict:foamL")
    buoyancy_mag_dict["foamR"] = get_param(r, "buoyancy_mag_dict:foamR")
    buoyancy_mag_dict["shoulder"] = get_param(r, "buoyancy_mag_dict:shoulder")
    buoyancy_mag_dict["upperarm"] = get_param(r, "buoyancy_mag_dict:upperarm")
    buoyancy_mag_dict["armbase"] = get_param(r, "buoyancy_mag_dict:armbase")
    buoyancy_mag_dict["jaw"] = get_param(r, "buoyancy_mag_dict:jaw")
    buoyancy_mag_dict["wrist"] = get_param(r, "buoyancy_mag_dict:wrist")
    buoyancy_mag_dict["elbow"] = get_param(r, "buoyancy_mag_dict:elbow")


    # Overwrite com_vec_dict
    com_vec_dict["vehicle"] = SVector{3, Float64}(
        get_param(r, "com_vec_dict:vehicle:0"),
        get_param(r, "com_vec_dict:vehicle:1"),
        get_param(r, "com_vec_dict:vehicle:2")
    )

    com_vec_dict["weightCA"] = SVector{3, Float64}(
        get_param(r, "com_vec_dict:weightCA:0"),
        get_param(r, "com_vec_dict:weightCA:1"),
        get_param(r, "com_vec_dict:weightCA:2")
    )

    com_vec_dict["weightBL"] = SVector{3, Float64}(
        get_param(r, "com_vec_dict:weightBL:0"),
        get_param(r, "com_vec_dict:weightBL:1"),
        get_param(r, "com_vec_dict:weightBL:2")
    )

    com_vec_dict["dvl"] = SVector{3, Float64}(
        get_param(r, "com_vec_dict:dvl:0"),
        get_param(r, "com_vec_dict:dvl:1"),
        get_param(r, "com_vec_dict:dvl:2")
    )

    com_vec_dict["dvlbracket"] = SVector{3, Float64}(
        get_param(r, "com_vec_dict:dvlbracket:0"),
        get_param(r, "com_vec_dict:dvlbracket:1"),
        get_param(r, "com_vec_dict:dvlbracket:2")
    )

    com_vec_dict["armbase"] = SVector{3, Float64}(
        get_param(r, "com_vec_dict:armbase:0"),
        get_param(r, "com_vec_dict:armbase:1"),
        get_param(r, "com_vec_dict:armbase:2")
    )

    com_vec_dict["shoulder"] = SVector{3, Float64}(
        get_param(r, "com_vec_dict:shoulder:0"),
        get_param(r, "com_vec_dict:shoulder:1"),
        get_param(r, "com_vec_dict:shoulder:2")
    )

    com_vec_dict["upperarm"] = SVector{3, Float64}(
        get_param(r, "com_vec_dict:upperarm:0"),
        get_param(r, "com_vec_dict:upperarm:1"),
        get_param(r, "com_vec_dict:upperarm:2")
    )

    com_vec_dict["elbow"] = SVector{3, Float64}(
        get_param(r, "com_vec_dict:elbow:0"),
        get_param(r, "com_vec_dict:elbow:1"),
        get_param(r, "com_vec_dict:elbow:2")
    )

    com_vec_dict["wrist"] = SVector{3, Float64}(
        get_param(r, "com_vec_dict:wrist:0"),
        get_param(r, "com_vec_dict:wrist:1"),
        get_param(r, "com_vec_dict:wrist:2")
    )

    com_vec_dict["jaw"] = SVector{3, Float64}(
        get_param(r, "com_vec_dict:jaw:0"),
        get_param(r, "com_vec_dict:jaw:1"),
        get_param(r, "com_vec_dict:jaw:2")
    )

    com_vec_dict["jaw_wrt_wrist"] = SVector{3, Float64}(
        get_param(r, "com_vec_dict:jaw_wrt_wrist:0"),
        get_param(r, "com_vec_dict:jaw_wrt_wrist:1"),
        get_param(r, "com_vec_dict:jaw_wrt_wrist:2")
    )

    # Overwrite grav_mag_dict
    grav_mag_dict["vehicle"] = get_param(r, "grav_mag_dict:vehicle")
    grav_mag_dict["weightCA"] = get_param(r, "grav_mag_dict:weightCA")
    grav_mag_dict["weightBL"] = get_param(r, "grav_mag_dict:weightBL")
    grav_mag_dict["weightBR"] = get_param(r, "grav_mag_dict:weightBR")
    grav_mag_dict["dvl"] = get_param(r, "grav_mag_dict:dvl")
    grav_mag_dict["dvlbracket"] = get_param(r, "grav_mag_dict:dvlbracket")
    grav_mag_dict["armbase"] = get_param(r, "grav_mag_dict:armbase")
    grav_mag_dict["shoulder"] = get_param(r, "grav_mag_dict:shoulder")
    grav_mag_dict["upperarm"] = get_param(r, "grav_mag_dict:upperarm")
    grav_mag_dict["elbow"] = get_param(r, "grav_mag_dict:elbow")
    grav_mag_dict["jaw"] = get_param(r, "grav_mag_dict:jaw")
    grav_mag_dict["wrist"] = get_param(r, "grav_mag_dict:wrist")

    # Overwrite drag parameters
    d_lin_angular = get_param(r, "drag:d_lin_angular")
    d_nonlin_angular = get_param(r, "drag:d_nonlin_angular")
    d_lin_coeffs = [
        get_param(r, "drag:d_lin_coeffs:0"),
        get_param(r, "drag:d_lin_coeffs:1"),
        get_param(r, "drag:d_lin_coeffs:2"),
        d_lin_angular,
        d_lin_angular,
        d_lin_angular
    ]
    d_nonlin_coeffs = [
        get_param(r, "drag:d_nonlin_coeffs:0"),
        get_param(r, "drag:d_nonlin_coeffs:1"),
        get_param(r, "drag:d_nonlin_coeffs:2"),
        d_nonlin_angular,
        d_nonlin_angular,
        d_nonlin_angular
    ]

    # Overwrite link volumes
    link_volumes["shoulder"] = get_param(r, "link_volumes:shoulder")
    link_volumes["upperarm"] = get_param(r, "link_volumes:upperarm")
    link_volumes["elbow"] = get_param(r, "link_volumes:elbow")
    link_volumes["wrist"] = get_param(r, "link_volumes:wrist")
    link_volumes["armbase"] = get_param(r, "link_volumes:armbase")
    link_volumes["jaw"] = get_param(r, "link_volumes:jaw")

    # Overwrite link masses
    link_masses["shoulder"] = get_param(r, "link_masses:shoulder")
    link_masses["upperarm"] = get_param(r, "link_masses:upperarm")
    link_masses["elbow"] = get_param(r, "link_masses:elbow")
    link_masses["wrist"] = get_param(r, "link_masses:wrist")
    link_masses["armbase"] = get_param(r, "link_masses:armbase")
    link_masses["jaw"] = get_param(r, "link_masses:jaw")

    # Overwrite link drags
    link_drags["shoulder"] = SVector{3, Float64}(
        get_param(r, "link_drags:shoulder:0"),
        get_param(r, "link_drags:shoulder:1"),
        get_param(r, "link_drags:shoulder:2")
    )

    link_drags["upperarm"] = SVector{3, Float64}(
        get_param(r, "link_drags:upperarm:0"),
        get_param(r, "link_drags:upperarm:1"),
        get_param(r, "link_drags:upperarm:2")
    )

    link_drags["elbow"] = SVector{3, Float64}(
        get_param(r, "link_drags:elbow:0"),
        get_param(r, "link_drags:elbow:1"),
        get_param(r, "link_drags:elbow:2")
    )

    link_drags["wrist"] = SVector{3, Float64}(
        get_param(r, "link_drags:wrist:0"),
        get_param(r, "link_drags:wrist:1"),
        get_param(r, "link_drags:wrist:2")
    )

    link_drags["jaw"] = SVector{3, Float64}(
        get_param(r, "link_drags:jaw:0"),
        get_param(r, "link_drags:jaw:1"),
        get_param(r, "link_drags:jaw:2")
    )

    # Redefine frames, forces, states
    cob_frame_dict, com_frame_dict = setup_frames(body_dict, body_names, cob_vec_dict, com_vec_dict)
    buoyancy_force_dict, gravity_force_dict = setup_buoyancy_and_gravity(buoyancy_mag_dict, grav_mag_dict)

    state = MechanismState(mech_blue_alpha)
    num_dofs = num_velocities(mech_blue_alpha)
    num_actuated_dofs = num_dofs-2
end