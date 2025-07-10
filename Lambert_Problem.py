import time
"""
Notes (Trey):
- changed .sec *u.s to .to(u.s) for more robust conversion

- could be made parallel or openMDAO to en
"""
"""
Wrap this around anything you want to time.

time1 = time.time()

time2 = time.time()
time_elapsed = time2-time1
print(f"Time elapsed: {time_elapsed}")

"""
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.constants import J2000
from poliastro.twobody.orbit import Orbit
import poliastro.twobody.elements as elements
from poliastro.maneuver import Maneuver
import math



def nu_over_time(orbit, node_epochs, current_epoch):

    """
    Calculates the true anomaly of the client satellite when the MRRV reaches the target orbit.
    Inputs:
        orbit (Orbit)      - client satellite orbit
        node_epochs (list) - epochs for the client satellite's critical points (first pass only)
        current_epoch      - epoch when the MRRV reaches the target orbit

    Outputs:
        nu - true anomaly of the client satellite when the MRRV reaches the target orbit

    Order of critical points:
     1) perigee
     2) apogee
     3) ascending node
     4) descending node
    """

    global time_in_nu_over_time, nu_over_time_calls
    nu_over_time_calls += 1
    call_time = time.time()

    T = node_epochs[0]
    dt = (current_epoch - T).to(u.s)
    ecc = orbit.ecc.value
    period = orbit.period
    M = 2*math.pi*dt/period * u.rad
    M = math.fmod(M.value, 2*math.pi) * u.rad

    E0 = math.pi/2 * u.rad
    diff = 1
    tol = 1e-12
    j = 1
    while abs(diff) > tol:
        F = E0 - ecc*math.sin(E0.value)*u.rad - M
        dFdE = 1 - ecc*math.cos(E0.value)
        E1 = E0 - F/dFdE
        diff = (E1 - E0).value
        j = j+1
        E0 = E1
    
    E = (E1.value % (2*math.pi)) * u.rad
    nu = 2*math.atan(math.tan(E.value/2)/math.sqrt((1-ecc)/(1+E.value)))

    run_time = time.time() - call_time
    time_in_nu_over_time += run_time

    return (nu % (2*math.pi)) * u.rad

def time_to_nodes(orbit):

    """
    Calculates the travel time between the satellite's current position and each critical point in its orbit.
    Inputs:
        orbit (Orbit) - satellite orbit defined at its current position

    Outputs:
        dt_vec (list) - time to reach critical points
        nu_vec (list) - true anomaly of critical points

    Order of critical points:
     1) perigee
     2) apogee
     3) ascending node
     4) descending node
    """

    global time_in_time_to_nodes, time_to_nodes_calls
    time_to_nodes_calls += 1
    call_time = time.time()

    argp = orbit.argp.value
    ecc = orbit.ecc.value
    period = orbit.period.value
    nu_i = orbit.nu.value

    nu_vec = (0, math.pi, 2*math.pi-argp, math.pi-argp)
    dt_vec = []
    E1 = math.acos((ecc+math.cos(nu_i))/(1+ecc*math.cos(nu_i)))
    M1 = E1 - ecc * math.sin(E1)
    for nu in nu_vec:
        E2 = math.acos((ecc+math.cos(nu))/(1+ecc*math.cos(nu)))
        M2 = E2 - ecc * math.sin(E2)
        dM = M2 - M1
        k = 0
        dt = -1
        while dt < 0:
            dt = (period/2/math.pi)*(2*k*math.pi + dM) * u.s
            k = k + 1
        dt_vec.append(dt)

    run_time = time.time() - call_time
    time_in_time_to_nodes += run_time

    return dt_vec, nu_vec

def define_orbit_nodes(orbit, node_epochs, nu_vec):

    """
    Creates Orbit objects at each critical point at the appropriate epoch.
    Inputs:
        orbit (Orbit)       - satellite orbit defined at its starting position and epoch
        node_epochs (list)  - epochs for the satellite's critical points (first pass only)
        nu_vec (list)       - true anomaly of critical points

    Outputs:
        orbit_nodes (list)  - Orbit objects for each critical point at the appropriate epoch

    Order of critical points:
     1) perigee
     2) apogee
     3) ascending node
     4) descending node
    """

    global time_in_define_orbit_nodes, define_orbit_nodes_calls
    define_orbit_nodes_calls += 1
    call_time = time.time()

    orbit_nodes = []
    k = Earth.k.value / 1e9 * u.km * u.km * u.km / u.s / u.s
    for node_epoch, nu in zip(node_epochs, nu_vec):
        # Finding r and v at each critical point is necessary to create an Orbit object for that point
        r, v = elements.coe2rv(k=k, p=orbit.p, ecc=orbit.ecc, inc=orbit.inc, raan=orbit.raan, argp=orbit.argp, nu=nu*u.rad)
        orbit_node = Orbit.from_vectors(Earth, r, v, epoch=node_epoch)
        orbit_nodes.append(orbit_node)

    run_time = time.time() - call_time
    time_in_define_orbit_nodes += run_time

    return orbit_nodes

def phasing_orbit(orbit, client_node_epoch, dV_constraint):

    """
    Calculates the lowest-TOF set of phasing maneuvers for bringing the MRRV into phase
    with the client satellite, subject to a delta V constraint.

    Inputs:
        orbit (Orbit) - MRRV orbit when it reaches the target orbit (target orbit, not transfer orbit)
        client_node_epoch - epoch when the client satellite reaches the critical point
        dV_constraint - maximum delta V available for phasing

    Outputs:
        optimal_phasing_burn (Maneuver)    - maneuver for the initial phasing burn (time = 0)
        optimal_rendezvous_burn (Maneuver) - maneuver for the burn bringing MRRV into RPO w/ client satellite (time = duration of phasing)
    """

    global time_in_phasing_orbit, phasing_orbit_calls, phasing_orbit_longest_runtime
    phasing_orbit_calls += 1
    call_time = time.time()

    mu = Earth.k.to(u.km**3 / u.s**2)
    mrrv_arrival_epoch = orbit.epoch
    period = orbit.period
    r = orbit.r
    v = orbit.v
    r_mag = math.sqrt(sum(component.value**2 for component in r)) * u.km
    v_mag = math.sqrt(sum(component.value**2 for component in v)) * u.km / u.s
    nearest_client_node_epoch = None
    i = 0
    while nearest_client_node_epoch is None:
        if client_node_epoch + i*period > mrrv_arrival_epoch:
            nearest_client_node_epoch = client_node_epoch + (i-1)*period
        i = i+1

    fallback_time = (mrrv_arrival_epoch - nearest_client_node_epoch).to(u.s)

    # Calculate fallback costs
    n_orbits = 0
    dV_phasing_total = dV_constraint + 1 * u.km / u.s
    while dV_phasing_total > dV_constraint:
        n_orbits = n_orbits + 1
        fallback_per_orbit = fallback_time / n_orbits
        period_phasing = period + fallback_per_orbit
        a_phasing = (mu * (period_phasing/(2*math.pi))**2)**(1/3)
        v_phasing_mag = math.sqrt((mu * (2/r_mag - 1/a_phasing)).value) * u.km / u.s
        dV_phasing_total = 2 * abs(v_phasing_mag.value - v_mag.value) * u.km / u.s
    optimal_fallback_v = [(component.value / v_mag.value) * v_phasing_mag.value for component in v] * u.km / u.s
    optimal_fallback_dV = optimal_fallback_v - v
    optimal_fallback_phasing_burn = Maneuver.impulse(optimal_fallback_dV)
    optimal_fallback_rendezvous_burn = Maneuver.impulse(-1 * optimal_fallback_dV)
    optimal_fallback_rendezvous_burn._dts[0] = period_phasing*n_orbits

    catchup_time = abs((nearest_client_node_epoch - mrrv_arrival_epoch)).to(u.s)

    # Calculate catchup costs
    n_orbits = 0
    dV_phasing_total = dV_constraint + 1 * u.km / u.s
    while dV_phasing_total > dV_constraint:
        n_orbits = n_orbits + 1
        catchup_per_orbit = catchup_time / n_orbits
        period_phasing = period - catchup_per_orbit
        a_phasing = (mu * (period_phasing/(2*math.pi))**2)**(1/3)
        try:
            v_phasing_mag = math.sqrt((mu * (2/r_mag - 1/a_phasing)).value) * u.km / u.s
        except:
            continue
        dV_phasing_total = 2 * abs(v_phasing_mag.value - v_mag.value) * u.km / u.s
    optimal_catchup_v = [(component.value / v_mag.value) * v_phasing_mag.value for component in v] * u.km / u.s
    optimal_catchup_dV = optimal_catchup_v - v
    optimal_catchup_phasing_burn = Maneuver.impulse(optimal_catchup_dV)
    optimal_catchup_rendezvous_burn = Maneuver.impulse(-1 * optimal_catchup_dV)
    optimal_catchup_rendezvous_burn._dts[0] = period_phasing*n_orbits

    if optimal_fallback_rendezvous_burn._dts[0] < optimal_catchup_rendezvous_burn._dts[0]:
        optimal_phasing_burn = optimal_fallback_phasing_burn
        optimal_rendezvous_burn = optimal_fallback_rendezvous_burn
    else:
        optimal_phasing_burn = optimal_catchup_phasing_burn
        optimal_rendezvous_burn = optimal_catchup_rendezvous_burn

    run_time = time.time() - call_time
    time_in_phasing_orbit += run_time
    phasing_orbit_longest_runtime = max(run_time, phasing_orbit_longest_runtime)

    return optimal_phasing_burn, optimal_rendezvous_burn

def fastest_route(initial_orbit, r, v, dV_constraint):

    """
    Calculates the set of maneuvers that will bring MRRV into RPO with a client satellite,
    subject to some delta V constraint.

    Inputs:
        initial_orbit (Orbit) - MRRV's starting orbit
        r (list)              - initial position vector for the client satellite
        v (list)              - initial velocity vector for the client satellite
        dV_constraint         - maximum allowable delta V for the entire set of maneuvers
    
    Outputs:
        optimal_flight (list) - fastest set of feasible maneuvers
    """

    time_initial = time.time() 

    # Get critical point epochs and true anomalies for the initial and target orbits
    starting_epoch = initial_orbit.epoch
    target_orbit = Orbit.from_vectors(Earth, r, v, epoch=starting_epoch)
    initial_dts, initial_nu_vec = time_to_nodes(initial_orbit)
    target_dts, target_nu_vec = time_to_nodes(target_orbit)

    initial_epochs = []
    target_epochs = []
    for initial_travel_time, target_travel_time in zip(initial_dts, target_dts):
        initial_epochs.append(starting_epoch + initial_travel_time)
        target_epochs.append(starting_epoch + target_travel_time)

    # Use critical point epochs and true anomalies to create Orbit objects for each point
    initial_orbit_nodes = define_orbit_nodes(initial_orbit, initial_epochs, initial_nu_vec)
    target_orbit_nodes = define_orbit_nodes(target_orbit, target_epochs, target_nu_vec)

    # Core loop
    flight_options = []
    for i, initial_orbit_node in enumerate(initial_orbit_nodes):
        print(f"Running critical point {i+1}...")
        departure_node_epoch = initial_orbit_node.epoch
        departure_node_options = []
        for j, target_orbit_node in enumerate(target_orbit_nodes):
            r = target_orbit_node.r
            v = target_orbit_node.v

            # Initial guess for TOF is set to 10 seconds
            tof_transfer = 10 * u.s

            # Increasing delta increases the coarseness of the iterations.  Increasing delta reduces runtime,
            # but hurts performance, and may lead to feasible flights not being found.
            delta = 100 * u.s

            # Initializing the loop
            dV_transfer_min = 9e9 * u.km / u.s
            dV_transfer = dV_transfer_min
            total_tof_min = 9e9 * u.s
            total_tof = total_tof_min
            loop = 0

            max_tof = 86400 * u.s
            transfer_found = False
            # Stop looping if delta V and total TOF go up with iterations
            while tof_transfer < total_tof_min and tof_transfer < max_tof:
                loop = loop + 1
                dV_transfer_min = min([dV_transfer_min.value, dV_transfer.value]) * u.km / u.s
                total_tof_min = min([total_tof_min.value, total_tof.value]) * u.s
                arrival_epoch = departure_node_epoch + tof_transfer
                target_orbit_node = Orbit.from_vectors(Earth, r, v, epoch=arrival_epoch)
                transfer = Maneuver.lambert(initial_orbit_node, target_orbit_node)
                transfer._dts[0] = (departure_node_epoch - starting_epoch).value * 86400 * u.s
                transfer._dts[1] = transfer._dts[0] + transfer._dts[1]
                dV_transfer = transfer.get_total_cost()

                # Multiplying the dV constraint by some number < 1 reduces the likelihood that phasing_orbit
                # will run for an extended period.  This may lead to feasible flights not being found.
                if dV_transfer > dV_constraint * 0.9:
                    # Cost is too high, so we increase TOF and run again
                    tof_transfer = tof_transfer + delta
                else:
                    # Cost is lower that dV_constraint
                    transfer_found = True
                    # client_nu_at_arrival = nu_over_time(target_orbit, target_epochs, arrival_epoch)
                    phasing_dV_constraint = dV_constraint - dV_transfer
                    phasing_burn, rendezvous_burn = phasing_orbit(target_orbit_node, target_epochs[j], dV_constraint=phasing_dV_constraint)
                    phasing_burn._dts[0] = transfer._dts[1]
                    rendezvous_burn._dts[0] = rendezvous_burn._dts[0] + phasing_burn._dts[0]
                    total_tof = rendezvous_burn._dts[0]
                    tof_transfer = tof_transfer + delta
                    if total_tof < total_tof_min:
                        optimal_transfer = transfer
                        optimal_phasing_burn = phasing_burn
                        optimal_rendezvous_burn = rendezvous_burn
                        total_tof_min = total_tof
                                    
            if transfer_found:    
                flight = [optimal_transfer, optimal_phasing_burn, optimal_rendezvous_burn]
            else:
                flight = [False, False, False]
            departure_node_options.append(flight)
        flight_options.append(departure_node_options)
        

    optimal_flight_tof = 9e9 * u.s
    for departure_node_options in flight_options:
        for flight in departure_node_options:
            if flight[2] is not False:
                if flight[2]._dts[0] < optimal_flight_tof:
                    optimal_flight = flight
                    optimal_flight_tof = flight[2]._dts[0]

    print(f"Optimal TOF: {optimal_flight_tof:.2f}")
    time_final = time.time()
    time_elapsed = time_final-time_initial
    print(f"Runtime: {time_elapsed:.2f} s")
    
    return optimal_flight

# time_in_nu_over_time = 0
time_in_time_to_nodes = 0
time_in_define_orbit_nodes = 0
time_in_phasing_orbit = 0

# nu_over_time_calls = 0
time_to_nodes_calls = 0
define_orbit_nodes_calls = 0
phasing_orbit_calls = 0

phasing_orbit_longest_runtime = 0

r1 = [8000, 0, 0] * u.km  # Initial position vector
r2 = [8000, -1000, 2000] * u.km  # Target position vector

v1 = [1, 8, 0] * u.km / u.s # Initial velocity vector
v2 = [2, 7, 0] * u.km / u.s # Target velocity vector

starting_epoch = J2000

initial_orbit = Orbit.from_vectors(Earth, r1, v1, epoch=starting_epoch)
target_orbit = Orbit.from_vectors(Earth, r2, v2, epoch=starting_epoch)

dV_constraint = 100 * u.km / u.s
optimal_flight = fastest_route(initial_orbit, r2, v2, dV_constraint)

# avg_runtime_nu_over_time = time_in_nu_over_time / nu_over_time_calls
avg_runtime_time_to_nodes = time_in_time_to_nodes / time_to_nodes_calls
avg_runtime_define_orbit_nodes = time_in_define_orbit_nodes / define_orbit_nodes_calls
avg_runtime_phasing_orbit = time_in_phasing_orbit / phasing_orbit_calls

print(f"delta V constraint: {dV_constraint}")
print(optimal_flight)
print("\n")


# print(" ----------- nu_over_time ----------- ")
# print(f"Calls: {nu_over_time_calls}")
# print(f"Total time: {time_in_nu_over_time:.2f} s")
# print(f"Average runtime: {avg_runtime_nu_over_time:.4f} s")
# print("\n")

print(" ----------- time_to_nodes ----------- ")
print(f"Calls: {time_to_nodes_calls}")
print(f"Total time: {time_in_time_to_nodes:.2f} s")
print(f"Average runtime: {avg_runtime_time_to_nodes:.4f} s \n")

print(" ----------- define_orbit_nodes ----------- ")
print(f"Calls: {define_orbit_nodes_calls}")
print(f"Total time: {time_in_define_orbit_nodes:.2f} s")
print(f"Average runtime: {avg_runtime_define_orbit_nodes:.4f} s \n")

print(" ----------- phasing_orbit ----------- ")
print(f"Calls: {phasing_orbit_calls}")
print(f"Total time: {time_in_phasing_orbit:.2f} s")
print(f"Average runtime: {avg_runtime_phasing_orbit:.4f} s")
print(f"Longest runtime: {phasing_orbit_longest_runtime:.2f} s")

######## for accessing the times and burns for the simulation portion... 
all_times = []
all_dvs = []

for burn in optimal_flight:
      for impulse_index in range(len(burn._dts)):
        burn_times = burn._dts[impulse_index]
        dv = burn._dvs[impulse_index]

        all_times.append(burn_times)
        all_dvs.append(dv) 