def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    
    updated_values = []

    for s in range(len(values)):
        updated_s = []
        for a in range(len(rewards[0])):
            r = rewards[s][a]
            for sp in range(len(values)):
                updated_s.append(r + gamma * transitions[s][a][sp] * values[sp])
        
        updated_values.append(max(updated_s))

    return updated_values