using RigidBodyDynamics.OdeIntegrators
import RigidBodyDynamics: default_constraint_stabilization_gains, MechanismState
import RigidBodyDynamics
import RigidBodyDynamics.cache_eltype

function simulate_with_ext_forces(actuator_limits, pitch_pred_pars, sensor_noise_dist, num_dofs, com_frame_dict, cob_frame_dict, const_magic_nums, magic_num_pitch_val, magic_num_bluerov_with_alpha_arm, body_dict, joint_dict, state0::MechanismState{X}, final_time, pars, ctlr, hydro_calc!, mvis, control! = zero_torque!;
        stabilization_gains=default_constraint_stabilization_gains(X)) where X
    # println("Made it to the simulate function!")
        T = cache_eltype(state0)
    result = DynamicsResult{T}(state0.mechanism)
    control_torques = similar(velocity(state0))
    hydro_wrenches = Dict{BodyID, Wrench{Float64}}()
    closed_loop_dynamics! = let result=result, hydro_wrenches=hydro_wrenches, control_torques=control_torques, stabilization_gains=stabilization_gains # https://github.com/JuliaLang/julia/issues/15276
        function (v̇::AbstractArray, ṡ::AbstractArray, t, state)
            # println("------ NEW SIM STATE -----")
            # println("Current State:")
            # println(configuration(state))
            hydro_calc!(com_frame_dict, cob_frame_dict, const_magic_nums, magic_num_pitch_val, magic_num_bluerov_with_alpha_arm, body_dict, joint_dict, hydro_wrenches, t, state, mvis)
            # println("Hydro wrenches")
            # println(hydro_wrenches)

            control!(actuator_limits, pitch_pred_pars, sensor_noise_dist, num_dofs, const_magic_nums, magic_num_pitch_val, control_torques, t, state, pars, ctlr, result, hydro_wrenches)
            # println("Control torques")
            # println(control_torques)
            # println("--------------- NEW ITERATION -------------------")
            # println("prev result")
            # println(result)

            # println("State")
            # println(configuration(state))
            # println(velocity(state))
            # println("Control torques:")
            # println(control_torques)
            # println("Hydro Wrenches")
            # println(hydro_wrenches)
            dynamics!(result, state, control_torques, hydro_wrenches; stabilization_gains=stabilization_gains)
            # println("result:")
            # println(result.v̇)
            copyto!(v̇, result.v̇)
            copyto!(ṡ, result.ṡ)
            nothing
        end
    end
tableau = runge_kutta_4(T)
storage = ExpandingStorage{T}(state0, ceil(Int64, final_time / const_magic_nums.Δt * 1.001)) # very rough overestimate of number of time steps
integrator = MuntheKaasIntegrator(state0, closed_loop_dynamics!, tableau, storage)
integrate(integrator, final_time, const_magic_nums.Δt)
storage.ts, storage.qs, storage.vs
end