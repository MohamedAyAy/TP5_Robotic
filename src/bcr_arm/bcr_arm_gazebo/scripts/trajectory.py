import numpy as np

def quintic_trajectory(q0, qf, T: float, n_points: int = 50):
    """Trajectoire quintique 7D entre q0 et qf sur durée T."""
    q0 = np.asarray(q0, dtype=float)
    qf = np.asarray(qf, dtype=float)
    times = np.linspace(0.0, T, n_points)
    
    dq = qf - q0
    
    # Coefficients du polynôme quintique : a3*t^3 + a4*t^4 + a5*t^5
    a3 = 10.0 * dq / T**3
    a4 = -15.0 * dq / T**4
    a5 = 6.0 * dq / T**5
    
    positions = np.array([
        q0 + a3 * (t**3) + a4 * (t**4) + a5 * (t**5)
        for t in times
    ])
    
    return times, positions
