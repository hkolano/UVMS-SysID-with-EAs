function evaluate_parameter(p)
    # Initialize internal state
    state = 0
    # Add parameter to that state 
    # (as if taking an action multiple times conditioned on that parameter)
    for _ in 1:5
        state = state + p
    end
    # Set the target state we are trying to get to
    target = 20
    # Evaluate that state that we got to. Julia implicitly returns the output of the last line
    abs(target-state)
end
