"""
Notes: 
This version of DES processes has only Large client satellites, all instances of small clients have been removed

adapted to test out the fuel types and new autopopulated dry mass
========================================

"""
import textwrap
import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
#from Object_Definition import Client, MRRV, Depot
from Object_Definition_Fuel_Types import Client, MRRV, Depot
from helper_functions import *
import copy
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import truncnorm

# Constants
g0 = 0.0098 * u.km / (u.s**2)
global_seed = 1

# ==================================================== DES Processes Definition ====================================================
def refuel_request(env, request_queue, result_moes):
    """
    Handles refuel requests from both client satellites and depots.
    """
    while True:
        # Get the next refuel request
        requester, request = yield request_queue.get()
        dt = env.now * u.s - requester.orbit.epoch
        requester = time_to_true_anomaly(requester, dt)

        # Check if the requester is a client satellite or a depot
        if isinstance(requester, Client):
            # Handle client satellite refuel request
            #print(f"t0 = {(env.now * u.s).to(u.day):.3f}: {requester.name} is requesting refuel.")
            requester.state = "Waiting"  # Mark as waiting for service

            # Get the best MRRV for the refuel request
            best_mrrv, best_times, best_dvs = yield from closest_check(env, requester, mrrv_satellites, request)

            # Ensure valid return values
            if not best_mrrv or best_times is None or best_dvs is None:
                #print(f"No suitable MRRV found for {requester.name}. Requeuing request...")
                request_queue.put((requester, request))
                continue  # Skip to next loop iteration

            #print(f"{best_mrrv.name} assigned to refuel {requester.name}.")
            requester.state = "Being Serviced"
            best_mrrv.state = "Busy"

            # Proceed with refueling
            env.process(handle_refueling(env, best_mrrv, requester, request, result_moes, best_times, best_dvs))

        elif isinstance(requester, Depot):
            # Handle depot refuel request
            #print(f"t0 = {(env.now * u.s).to(u.day):.3f}: {requester.name} is requesting refuel. Simulating 7-day refueling mission from Earth...")
            requester.state = "Being Refueled"

            # Simulate a 7-day delay for the refueling mission
            yield env.timeout(7 * 86400)  # 7 days in seconds

            # Refill the depot to its maximum capacity
            requester.mass_prop = requester.mass_prop_max
            requester.state = "Available"
            #print(f"t1 = {(env.now * u.s).to(u.day):.3f}: {requester.name} has been refueled to {requester.mass_prop:.2f}.")


def handle_refueling(env, best_mrrv, client_sat, request, result_moes, best_times, best_dvs):
    """
    Handles a single MRRV's refueling mission in parallel.
    """
    start_mass_mrrv = best_mrrv.mass_prop
    t1 = env.now * u.s
    #print(f"t1 = {t1.to(u.day):.3f}: {best_mrrv.name} begins its flight to {client_sat.name}.")
    depot_satellites[f"{best_mrrv.home_depot}"].current_no_mrrvs -= 1 # Removing a mrrv from the depot parking space

    # Outward Flight
    best_mrrv_propagated = copy.deepcopy(best_mrrv)
    client_sat_propagated = copy.deepcopy(client_sat)
    yield env.process(execute_maneuver(env, best_mrrv_propagated, [best_times[0]], best_dvs[0]))
    yield env.process(execute_maneuver(env, best_mrrv, best_times, best_dvs))
    t2 = env.now * u.s
    dt2 = t2 - client_sat.orbit.epoch
    client_sat = time_to_true_anomaly(client_sat, dt2)  # ERROR? We expect true anomalies to match after this point...
    #print(f"t2 = {t2.to(u.day):.2f}: {best_mrrv.name} finishes its flight to {client_sat.name}. Now commencing RPO.")

    # RPO
    phasing_tof = best_times[-1] - best_times[0]
    best_mrrv_propagated.orbit = best_mrrv_propagated.orbit.propagate(phasing_tof, method=CowellPropagator(f=f))  # Need to ask logic of the propagator.. does it assume mrrv position hasn't been updated since before its maneuvers?
    client_sat_propagated.orbit = client_sat_propagated.orbit.propagate(dt2, method=CowellPropagator(f=f))
    best_mrrv, client_sat, rpo_time = yield env.process(rpo(env, best_mrrv, client_sat, client_sat_propagated, propagated_mobile_sat=best_mrrv_propagated))
    t3 = env.now * u.s
    #print(f"t3 = {t3.to(u.day):.3f}: {best_mrrv.name} docks with {client_sat.name} & begins refueling.")

    # Fuel Transfer
    _, _, _ = yield env.process(fuel_transfer(env, best_mrrv, client_sat, request))
    t4 = env.now * u.s
    #print(f"t4 = {t4.to(u.day):.3f}: {best_mrrv.name} ends fuel transfer to {client_sat.name}.")

    time_in_maintenance = (t4 - t1)
    client_sat.time_in_maintenance += time_in_maintenance
    client_sat.longest_maintenance_period = max([client_sat.longest_maintenance_period, time_in_maintenance])

    # If the client satellite was in the "down" state, update the total down time
    if client_sat.down_start_time is not None:
        down_end_time = env.now * u.s
        instance_down_time = down_end_time - client_sat.down_start_time
        client_sat.total_down_time += instance_down_time
        client_sat.longest_down_time_period = max([client_sat.longest_down_time_period, instance_down_time])
        client_sat.down_start_time = None  # Reset the down start time
        #print(f"[{(env.now * u.s).to(u.day):.3f}] {client_sat.name} is no longer DOWN. Total down time: {client_sat.total_down_time.to(u.day):.3f}")

    # Update client satellite state
    client_sat.state = "Available"  # Mark as available after servicing

    # Return Flight
    best_mrrv.state = "Returning"
    t5 = env.now * u.s
    depot, best_times, best_dvs = yield from closest_return(best_mrrv, depot_satellites)
    #print(f"t5 = {t5.to(u.day):.3f}: {best_mrrv.name} begins its flight to {depot.name}.")
    depot.current_no_mrrvs += 1 #updating the parking space count of depot
    best_mrrv.home_depot=depot.name
    best_mrrv_propagated = copy.deepcopy(best_mrrv)
    yield env.process(execute_maneuver(env, best_mrrv_propagated, [best_times[0]], best_dvs[0]))
    yield env.process(execute_maneuver(env, best_mrrv, best_times, best_dvs))
    t6 = env.now * u.s
    #print(f"t6 = {t6.to(u.day):.3f}: {best_mrrv.name} ends return flight & begins RPO with {depot.name}.")

    # RPO Return
    phasing_tof = best_times[-1] - best_times[0]
    depot_propagated = copy.deepcopy(depot)
    epoch_diff = depot_propagated.orbit.epoch - best_mrrv_propagated.orbit.epoch
    best_mrrv_propagated.orbit = best_mrrv_propagated.orbit.propagate(phasing_tof, method=CowellPropagator(f=f))
    depot_propagated.orbit = depot.orbit.propagate(phasing_tof - epoch_diff, method=CowellPropagator(f=f))
    depot = time_to_true_anomaly(depot, phasing_tof - epoch_diff)
    best_mrrv, depot, rpo_time = yield env.process(rpo(env, best_mrrv, depot, depot_propagated, propagated_mobile_sat=best_mrrv_propagated))
    t7 = env.now * u.s
    #print(f"t7 = {t7.to(u.day):.3f}: {best_mrrv.name} docks with {depot.name} & begins refueling.")

    # Fuel Transfer for MRRV
    fuel_needed = start_mass_mrrv - best_mrrv.mass_prop
    _, _, _ = yield env.process(fuel_transfer(env, depot, best_mrrv, None))
    best_mrrv.state = "Available"
    t8 = env.now * u.s
    #print(f"t8 = {t8.to(u.day):.3f}: {best_mrrv.name} refuel complete & now {best_mrrv.state}.")

    # Calculate MoEs
    total_time = (t8 - t1).to(u.day)
    total_fuel_expended = fuel_needed

    #print(f"======== {best_mrrv.name} mission complete =========")
    #print(f"Time to mission fulfillment = {total_time:.2f}")
    #print(f"Total fuel expended = {total_fuel_expended:.2f}")
    #print(f"==================================================")

    result_moes.append((total_time, total_fuel_expended))


def closest_check(env, client_sat, mrrv_list, request):
    """Select the best MRRV for a refuel request. If none are available, wait until one becomes available."""
    available_mrrvs = [m for m in mrrv_list.values() if m.state == "Available"]
    
    # If no MRRVs are available, wait until one becomes available
    while not available_mrrvs:
        #print(f"No MRRVs available. {client_sat.name} is waiting...")
        yield env.timeout(10000)  # Wait for a short time before checking again, could be better
        available_mrrvs = [m for m in mrrv_list.values() if m.state == "Available"]
        if available_mrrvs:
            dt = env.now * u.s - available_mrrvs[0].orbit.epoch
            client_sat = time_to_true_anomaly(client_sat, available_mrrvs[0].orbit.epoch - client_sat.orbit.epoch + dt)

    result_time_vec, result_dv_vec, result_times, result_dvs = [], [], [], []
    for mrrv in available_mrrvs:
        dt = env.now * u.s - mrrv.orbit.epoch
        mrrv = time_to_true_anomaly(mrrv, dt)
        _, dV_constraint_initial, _ = calculate_satellite_deltaVs(mrrv, request)
        optimal_initial_flight = calculate_phasing_orbit(mrrv, client_sat, dV_constraint_initial)
        if not optimal_initial_flight:
            continue

        all_times, all_dvs = [], []
        for burn in optimal_initial_flight:
            all_times.extend(burn._dts)
            all_dvs.extend(burn._dvs)

        result_time_vec.append(all_times)
        result_dv_vec.append(all_dvs)
        result_times.append(sum(all_times))
        result_dvs.append(sum(all_dvs))

    min_index = result_times.index(min(result_times))
    best_mrrv = available_mrrvs[min_index]
    best_mrrv.state = "Busy"
    return best_mrrv, result_time_vec[min_index], result_dv_vec[min_index]


def closest_return(mrrv, depot_list):
    """Select the best depot for MRRV return."""
    available_depots = [m for m in depot_list.values() if m.current_no_mrrvs < m.max_parking_spaces and m.state == 'Available']
    while not available_depots:
        #print(f"All Depots are full! {mrrv.name} is waiting...")
        yield env.timeout(10000)  # Wait for a short time before checking again, could be better
        available_depots = [m for m in depot_list.values() if m.current_no_mrrvs < m.max_parking_spaces and m.state == 'Available']

    result_time_vec, result_dv_vec, result_times, result_dvs = [], [], [], []
    for depot in available_depots:
        dt = env.now * u.s - depot.orbit.epoch
        depot = time_to_true_anomaly(depot, dt)
        request_return = {"refuel request amount": 0, "Percent dV allocated to initial flight": 0.9}
        
        # Debug: Check if calculate_satellite_deltaVs returns None
        deltaV_result = calculate_satellite_deltaVs(mrrv, request_return)
        if deltaV_result is None:
            #print(f"calculate_satellite_deltaVs returned None for {mrrv.name} and {depot.name}")
            continue
        _, dV_constraint_initial, _ = deltaV_result

        # Debug: Check if calculate_phasing_orbit returns None
        optimal_return_flight = calculate_phasing_orbit(mrrv, depot, dV_constraint_initial)
        if not optimal_return_flight:
            #print(f"calculate_phasing_orbit returned None for {mrrv.name} and {depot.name}")
            continue

        all_times, all_dvs = [], []
        for burn in optimal_return_flight:
            all_times.extend(burn._dts)
            all_dvs.extend(burn._dvs)

        result_time_vec.append(all_times)
        result_dv_vec.append(all_dvs)
        result_times.append(sum(all_times))
        result_dvs.append(sum(all_dvs))

    if not result_times:
        #print("No valid return flights found.")
        return None, None, None

    min_index = result_times.index(min(result_times))
    best_depot = available_depots[min_index]
    #best_depot.state = "Full" #Add in parking space compopnent
    return best_depot, result_time_vec[min_index], result_dv_vec[min_index]


def execute_maneuver(env, mrrv, times, dvs):
    """Execute a maneuver sequence."""
    if len(times) > 1:
        for idx, (time_i, dv_i) in enumerate(zip(times, dvs)):
            yield env.timeout(time_i.value)
            mrrv = time_to_true_anomaly(mrrv, time_i) # Update mrrv location to time right before burn
            burn = Maneuver.impulse(dvs[idx])
            #print(f"{mrrv.name} performed burn {idx+1} of {burn.get_total_cost():.3f} at simulation time {(env.now*u.s).to(u.day):.2f}")
            mrrv.orbit = mrrv.orbit.apply_maneuver(burn)
            dv_single_burn = np.linalg.norm(dv_i)
            initial_wet_mass = mrrv.mass_dry + mrrv.mass_prop
            fuel_burned_single_burn = rocketEQbackwards(dv_single_burn, mrrv.I_sp, initial_wet_mass, is_initial=True)
            mrrv.mass_prop = mrrv.mass_prop - fuel_burned_single_burn
    else:
        time_i = times[0]
        yield env.timeout(time_i.value)
        mrrv = time_to_true_anomaly(mrrv, time_i) # Update mrrv location to time right before burn
        burn = Maneuver.impulse(dvs)
        mrrv.orbit = mrrv.orbit.apply_maneuver(burn)
        dv_single_burn = np.linalg.norm(dvs)
        initial_wet_mass = mrrv.mass_dry + mrrv.mass_prop
        fuel_burned_initial = rocketEQbackwards(dv_single_burn, mrrv.I_sp, initial_wet_mass, is_initial=True)
        mrrv.mass_prop = mrrv.mass_prop - fuel_burned_initial


def fuel_transfer(env, provider, consumer, request):
    """Simulate fuel transfer between provider and consumer."""
    fuel_transfer_rate = min(provider.fuel_transfer_rate_out, consumer.fuel_transfer_rate_in)
    propellant_mass_transfered = request["refuel request amount"] if request else consumer.mass_prop_max - consumer.mass_prop
    fuel_transfer_duration = propellant_mass_transfered / fuel_transfer_rate

    if fuel_transfer_duration.value < 0:
        raise KeyError('Something wack')

    yield env.timeout(fuel_transfer_duration.value)
    provider.mass_prop -= propellant_mass_transfered
    consumer.mass_prop += propellant_mass_transfered
    
    provider = time_to_true_anomaly(provider, fuel_transfer_duration)
    consumer = time_to_true_anomaly(consumer, fuel_transfer_duration)

    consumer.total_prop_received += propellant_mass_transfered
    consumer.number_of_refuels += 1

    return provider, consumer, fuel_transfer_duration


def rpo(env, mobile_sat, stationary_sat, propagated_stationary_sat, propagated_mobile_sat=None):
    """Simulate rendezvous proximity operations."""
    #print(f"Calculating RPO: {mobile_sat.name} and {stationary_sat.name}")
    if propagated_mobile_sat is None:
        distance_to_target = np.linalg.norm(propagated_stationary_sat.orbit.r - mobile_sat.orbit.r)
    else:
        if propagated_mobile_sat.orbit.epoch.jd - propagated_stationary_sat.orbit.epoch.jd > 0.1:
            distance_to_target = np.linalg.norm(propagated_stationary_sat.orbit.r - mobile_sat.orbit.r)
        else:
            distance_to_target = np.linalg.norm(propagated_stationary_sat.orbit.r - propagated_mobile_sat.orbit.r)
    base_RPO_time = 5 * 60 * u.s
    RPO_time = base_RPO_time * (1 + ((distance_to_target.value/5) ** (1/3)))

    
    RPO_deltaV = 0.001 * u.km / u.s + 2 * (distance_to_target / RPO_time)
    RPO_fuel_burned = rocketEQbackwards(RPO_deltaV, mobile_sat.I_sp, mobile_sat.mass_dry + mobile_sat.mass_prop, is_initial=True) # Set to false, need to allocate the dV needed in rpo within the initial constraint determination!
    # mobile_sat.mass_prop -= RPO_fuel_burned

    if RPO_time > 2000 * u.s:
        mobile_sat_error = np.linalg.norm(propagated_mobile_sat.orbit.r - mobile_sat.orbit.r)
        stationary_sat_error = np.linalg.norm(propagated_stationary_sat.orbit.r - stationary_sat.orbit.r)
        raise Exception("RPO greater than 1000 s.  Suspected propagation error")
    
    stationary_sat = time_to_true_anomaly(stationary_sat, RPO_time)
    mobile_sat.orbit = stationary_sat.orbit

    yield env.timeout(RPO_time.value)

    return mobile_sat, stationary_sat, RPO_time

def satellite_fuel_depletion(env, client_sat, client_sats, fuel_threshold, down_threshold, request_queue):
    """
    Simulate fuel depletion for a client satellite and trigger refuel requests when fuel is low.
    """
    t_last_fuel_burn = client_sat.time_last_burn  # Initializing this counter
    client_sat_id = int(client_sat.name.split("_")[1])

    if client_sat_id == 12:
        client_sat.burn_this_timestep = 0 * u.kg

    while True:
        # Deplete fuel over time
        if client_sat.state != "Down":
            
            time_step = (env.now - t_last_fuel_burn) * u.s
            if time_step == 0: 
                chance_of_burn = 0 
                station_keeping_prop_burn = 0 * u.kg
            else:
                time_steps_per_year = (1 * u.year) / time_step.to(u.year)
                avg_maneuvers_per_year = (client_sat.maneuver_regularity - 1) * (49/99) + 3
                chance_of_burn = (avg_maneuvers_per_year / time_steps_per_year).value
                station_keeping_deltaV_per_time_step = 0.005 / time_steps_per_year * u.km / u.s
                station_keeping_prop_burn = rocketEQbackwards(station_keeping_deltaV_per_time_step, client_sat.I_sp, client_sat.mass_dry + client_sat.mass_prop, True)
            
            rng = client_sat.rng
            random_number = rng.random()
            if client_sat_id <= 12:
                if random_number > chance_of_burn:
                    client_sat.mass_prop = max(client_sat.mass_prop - station_keeping_prop_burn, 0 * u.kg)
                    client_sat.burn_this_timestep = station_keeping_prop_burn
                else:
                    if client_sat_id == 12:
                        group_states = []
                        for _, sat in client_sats.items():
                            group_states.append(sat.state)
                        group_states = group_states[-5:]
                        if all(s == "Available" for s in group_states) is not True:
                            client_sat.burn_this_timestep = 0 * u.kg
                            # client_sat.mass_prop = max(client_sat.mass_prop - client_sat.burn_this_timestep, 0 * u.kg)
                            continue
                    avg_prop_burn_per_year = 0.8 * client_sat.mass_prop_max
                    avg_prop_burn_per_maneuver = avg_prop_burn_per_year / avg_maneuvers_per_year

                    # Ensure avg_prop_burn_per_maneuver_st_dev is positive
                    avg_prop_burn_per_maneuver_st_dev = max(0.25 * avg_prop_burn_per_maneuver, 0.01 * u.kg)

                    # Ensure scale is positive
                    if avg_prop_burn_per_maneuver_st_dev <= 0:
                        avg_prop_burn_per_maneuver_st_dev = 0.01 * u.kg  # Set a small positive value as a fallback

                    # Calculate bounds for truncnorm
                    burn_upper_bound = client_sat.mass_prop - (0.1 * client_sat.mass_prop_max)
                    burn_lower_bound = 0 * u.kg

                    a = (burn_lower_bound - avg_prop_burn_per_maneuver) / avg_prop_burn_per_maneuver_st_dev
                    b = (burn_upper_bound - avg_prop_burn_per_maneuver) / avg_prop_burn_per_maneuver_st_dev

                    # Ensure a and b are valid (a < b)
                    if a >= b:
                        a, b = -np.inf, np.inf  # Fallback to unbounded normal distribution

                    # Generate random value from truncated normal distribution
                    maneuver_prop_burn = truncnorm.rvs(a, b, loc=avg_prop_burn_per_maneuver, scale=avg_prop_burn_per_maneuver_st_dev, random_state=rng) * u.kg
                    client_sat.burn_this_timestep = maneuver_prop_burn + station_keeping_prop_burn


                    # Update client satellite's propellant mass
                    client_sat.mass_prop = max(client_sat.mass_prop - (station_keeping_prop_burn + maneuver_prop_burn), 0 * u.kg)
            elif client_sat_id > 12:
                prop_burn_this_timestep = client_sats["client_satellite_12"].burn_this_timestep
                client_sat.mass_prop = max(client_sat.mass_prop - prop_burn_this_timestep, 0 * u.kg)

        # Check if fuel is below the threshold and the satellite is not already being serviced
        if client_sat.mass_prop <= client_sat.mass_prop_max * fuel_threshold and client_sat.state == "Available":
            #print(f"t0 = {(env.now * u.s).to(u.day):.3f}: {client_sat.name} fuel level is low ({client_sat.mass_prop:.2f}). Requesting refuel!")
            client_sat.state = "Waiting"  # Mark as waiting for service
            request = {
                "refuel request amount": client_sat.mass_prop_max - client_sat.mass_prop,
                "Percent dV allocated to initial flight": 0.8,
                "refuel request time": env.now
            }
            request_queue.put((client_sat, request))  # Add refuel request to the queue

        # Check if fuel is below the down threshold
        if client_sat.mass_prop <= client_sat.mass_prop_max * down_threshold and client_sat.down_start_time is None:
            client_sat.down_start_time = env.now * u.s  # Record the start of the "down" state
            client_sat.state = "Down"
            #print(f"t_X = {(env.now * u.s).to(u.day):.2f}: {client_sat.name} is DOWN! Fuel level: {client_sat.mass_prop:.2f}")

        # Check if fuel is above the down threshold and the satellite was previously down
        if client_sat.mass_prop > client_sat.mass_prop_max * down_threshold and client_sat.down_start_time is not None:
            down_end_time = env.now * u.s
            downtime = down_end_time - client_sat.down_start_time
            client_sat.total_down_time += downtime  # Update total down time
            client_sat.down_start_time = None  # Reset the down start time
            #print(f"t_X1 = {(env.now * u.s).to(u.day):.3f}: {client_sat.name} is no longer DOWN. Total down time: {client_sat.total_down_time.to(u.day):.3f}")

            # Store downtime data for plotting
            if client_sat.name not in downtime_data:
                downtime_data[client_sat.name] = []
            downtime_data[client_sat.name].append(client_sat.total_down_time.to(u.day).value)

        # Wait for the next fuel depletion check
        t_last_fuel_burn = env.now
        client_sat.time_last_burn = t_last_fuel_burn
        yield env.timeout(86400*7)  # Check fuel levels every week (adjust as needed)

def depot_fuel_depletion(env, depot, fuel_threshold, stationkeeping_deltaV_per_year, request_queue):
    """
    Simulate fuel depletion for a depot due to stationkeeping and refueling MRRVs.
    """
    # Convert stationkeeping delta-V from m/s per year to m/s per second
    stationkeeping_deltaV_per_second = (stationkeeping_deltaV_per_year*u.year) / ((365 * 24 * 60 * 60) * u.s)

    # Initialize the last fuel burn time
    t_last_depot_burn = env.now
    
    while True:
        # Calculate fuel burned for stationkeeping
        time_step = (env.now - t_last_depot_burn)*u.s  # Time since last check in seconds
        stationkeeping_fuel_burned = rocketEQbackwards(
            stationkeeping_deltaV_per_second * time_step,
            depot.I_sp,
            depot.mass_dry + depot.mass_prop,
            is_initial=True
        )
        depot.mass_prop = max(depot.mass_prop - stationkeeping_fuel_burned, 0 * u.kg)

        # Update the last fuel check time
        t_last_depot_burn = env.now

        # Check if fuel is below the threshold and the depot is not already being serviced
        if depot.mass_prop <= depot.mass_prop_max * fuel_threshold and depot.state == "Available":  
            #print(f"t_X = {(env.now * u.s).to(u.day):.3f}: {depot.name} fuel level is low ({depot.mass_prop:.2f}). Requesting refuel!")
            depot.state = "Waiting"  # Mark as waiting for service
            request = {
                "refuel request amount": depot.mass_prop_max - depot.mass_prop,
                "Percent dV allocated to initial flight": 0.8,
                "refuel request time": env.now
            }
            request_queue.put((depot, request))  # Add refuel request to the queue

        # Wait for the next fuel depletion check
        yield env.timeout(86400*7)  # Check fuel levels every week (adjust as needed)

def define_objects(case, global_seed):

    number_of_depots = int(case[3])
    mrrvs_per_depot = int(case[4])
    mrrv_prop_capacity = case[2]
    client_maneuver_regularity = case[5]

    number_of_small_cs = 12
    # number_of_large_cs = 0
    
    # satellite_rngs = global_rng.spawn(number_of_small_cs)
    satellite_rngs = [np.random.default_rng(global_seed + i) for i in range(number_of_small_cs+1)]
    # number_of_mrrvs = number_of_depots * mrrvs_per_depot

    # Initialize client satellites, MRRVs, and depots
    def_a = 42164.124522177
    def_ecc = 0
    def_inc = 0
    def_RAAN = 0
    def_omega = 0
    def_nu = 0

    random.seed(global_seed)
    depot_starting_nu = 2*math.pi*random.random()

    # Create Client Satellites
    client_satellites = {}
    for i in range(number_of_small_cs):
        client_satellites[f"client_satellite_{i}"] = Client(
            name=f"ClientSat_{i}",
            #mass_dry=200 * u.kg,
            mass_prop=100 * u.kg,
            mass_prop_max=100 * u.kg,
            #I_sp=400 * u.s,
            fuel_transfer_rate_in=0.006161075 * u.kg / u.s,  # Taken from orbitFab data sheet
            fuel_transfer_rate_out=0.006161075 * u.kg / u.s,
            initial_coe={
                "semi_major_axis": def_a * u.km,
                "eccentricity": def_ecc * u.one,
                "inclination": def_inc * u.rad,
                "RAAN": def_RAAN * u.rad,
                "omega": def_omega * u.rad,
                "true_anomaly": ((2*np.pi*i)/number_of_small_cs + def_nu)  * u.rad
            },
            initial_rv=None,
            fuel_depletion_rate= (40/365) * u.kg / u.day,
            maneuver_regularity= client_maneuver_regularity,
            type = "Small Client Satellite",
            rng = satellite_rngs[i],
            fuel_type="hydrazine"
        )
    # for j in range(number_of_large_cs):
    #     client_satellites[f"client_satellite_{i+j+1}"] = Client(
    #         name=f"ClientSat_{i+j+1}",
    #         mass_dry=400 * u.kg,
    #         mass_prop=200 * u.kg,
    #         mass_prop_max=200 * u.kg,
    #         I_sp=400 * u.s,
    #         fuel_transfer_rate_in=0.006161075 * u.kg / u.s,  # Taken from orbitFab data sheet
    #         fuel_transfer_rate_out=0.006161075 * u.kg / u.s,
    #         initial_coe={
    #             "semi_major_axis": def_a * u.km,
    #             "eccentricity": def_ecc * u.one,
    #             "inclination": def_inc * u.rad,
    #             "RAAN": def_RAAN * u.rad,
    #             "omega": def_omega * u.rad,
    #             "true_anomaly": (2 * np.pi * i) / number_of_large_cs * u.rad
    #         },
    #         initial_rv=None,
    #         fuel_depletion_rate = (200/365) * u.kg / u.day,
    #         maneuver_regularity = client_maneuver_regularity,
    #         type = "Big Client Satellite",
    #         rng = satellite_rngs[-1]
    #     )

    depot_satellites = {}
    mrrv_satellites = {}
    for i in range(number_of_depots):
        nu = ((2*np.pi*i)/number_of_depots + depot_starting_nu)  * u.rad

        depot_satellites[f"depot_{i}"] = Depot(
            name=f"depot_{i}",
            mass_dry=2000 * u.kg,
            mass_prop=9e9 * u.kg,
            mass_prop_max=9e9 * u.kg,
            I_sp=300 * u.s,
            fuel_transfer_rate_in=0.006161075 * u.kg / u.s,  # Taken from orbitFab data sheet
            fuel_transfer_rate_out=0.006161075 * u.kg / u.s,
            initial_coe={
                "semi_major_axis": def_a * u.km,
                "eccentricity": def_ecc * u.one,
                "inclination": def_inc * u.rad,
                "RAAN": def_RAAN * u.rad,
                "omega": def_omega * u.rad,
                "true_anomaly": nu
            },
            initial_rv=None,
            current_no_mrrvs=mrrvs_per_depot,
            max_parking_spaces=mrrvs_per_depot
        )

        for j in range(mrrvs_per_depot):
            mrrv_satellites[f"mrrv_{(i*mrrvs_per_depot)+j}"] = MRRV(
                name=f"MRRV_{(i*mrrvs_per_depot)+j}",
                mass_dry=100 * u.kg,
                mass_prop=mrrv_prop_capacity * u.kg,
                mass_prop_max=mrrv_prop_capacity * u.kg,
                I_sp=300 * u.s,
                fuel_transfer_rate_in=0.006161075 * u.kg / u.s,  # Taken from orbitFab data sheet
                fuel_transfer_rate_out=0.006161075 * u.kg / u.s,
                initial_coe={
                    "semi_major_axis": def_a * u.km,
                    "eccentricity": def_ecc * u.one,
                    "inclination": def_inc * u.rad,
                    "RAAN": def_RAAN * u.rad,
                    "omega": def_omega * u.rad,
                    "true_anomaly": nu
                },
                initial_rv=None,
                deorbit_dV_needed=10 * u.km / u.s,
                home_depot=depot_satellites[f"depot_{i}"].name
            )
    



    return client_satellites, depot_satellites, mrrv_satellites

def monitor(env, machine, interval=1):
    while True:
        machine.history.append([round(env.now / 86400, 3), machine.state, machine.mass_prop])
        yield env.timeout(interval)

# ==================== Main Simulation Loop ===============================================

doe_filename = "DoE_total_cases.csv"
doe_dataframe = build_doe_dataframe(doe_filename)
use_cases = doe_dataframe.values.tolist()

results_avg_fulfillment_time = []
results_total_fuel_expended = []
results_avg_fuel_expended = []
down_time = []  # Initialize list to store average downtime for each use case

doe_results = []
doe_individual_client_results = []

i = 0
for case in use_cases:
    i += 1
    print(f"\n===== Running Use Case: {i} / {len(use_cases)} =====")

    try:
        env = simpy.Environment()

        client_satellites, depot_satellites, mrrv_satellites = define_objects(case, global_seed)

        downtime_data = {}

        # Create a SimPy queue to store refuel requests
        request_queue = simpy.Store(env)

        # Start fuel depletion processes for client satellites
        fuel_threshold = case[1]  # Refuel when fuel level is below X %
        down_threshold = 0.2  # Mark CS as down when fuel level is below 20%
        for client_sat in client_satellites.values():
            env.process(satellite_fuel_depletion(env, client_sat, client_satellites, fuel_threshold, down_threshold, request_queue))

        # Start depot fuel depletion processes
        stationkeeping_deltaV_per_year = 50 * (u.m/u.s) / u.year  # Stationkeeping delta-V per year avg of GEO
        for depot in depot_satellites.values():
            env.process(depot_fuel_depletion(env, depot, fuel_threshold, stationkeeping_deltaV_per_year, request_queue))

        # Start refuel request handling process
        result_moes = []
        env.process(refuel_request(env, request_queue, result_moes))

        simulation_duration = 60 * 60 * 24 * 365.25 * 7
        #Run simulation
        #print("Running simulation...")
        env.run(until=simulation_duration)  # Run for 7 years (adjust as needed)
        #print("\n Simulation complete. \n")

        #Calculate and store results
        total_times = u.Quantity([time for time, _ in result_moes])
        total_fuels_burned = u.Quantity([fuel for _, fuel in result_moes])
        results_avg_fulfillment_time.append(np.average(total_times.value) * total_times.unit)
        results_total_fuel_expended.append(sum(total_fuels_burned))
        results_avg_fuel_expended.append(np.average(total_fuels_burned))

        # Calculate average downtime for this use case
        if downtime_data:
            avg_downtime = np.mean([sum(downtime) for downtime in downtime_data.values()])
        else:
            avg_downtime = 0
        down_time.append(avg_downtime)

        useless_metrics, useful_metrics = extract_metrics(simulation_duration, client_satellites, depot_satellites, mrrv_satellites)

        case_results = sum([case, useful_metrics], [])
        case_individual_client_results = sum([case, useless_metrics], [])
        
        write_results_to_csv(case_results, doe_filename, append="results", print_status_updates=False)
        write_results_to_csv(case_individual_client_results, doe_filename, append="individual_results", print_status_updates=False)

    except Exception as e:
        print(f"Case {i} FAILED")
        try:
            write_results_to_csv([], doe_filename, append="results", fail=True)
            write_results_to_csv([], doe_filename, append="individual_results", fail=True)
        except:
            continue
        continue