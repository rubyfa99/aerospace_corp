from astropy import units as u
from poliastro.bodies import Earth
from poliastro.constants import J2000
from poliastro.twobody.orbit import Orbit
import poliastro.twobody.elements as elements
from poliastro.maneuver import Maneuver
from poliastro.core.perturbations import J2_perturbation
from poliastro.twobody.propagation import cowell
from poliastro.twobody.propagation import CowellPropagator
from poliastro.core.propagation import func_twobody
import random
import numpy as np
import os
import pandas as pd
import csv

import copy
import math
# from Object_Definition_DSO_Rev2 import MRRV, Client

g0 = 9.81 * u.m / u.s**2

def time_to_true_anomaly(satellite, dt): # updated to Review 5 prototype 

    """
    Calculates the true anomaly of a satellite given some time-of-flight

    Inputs:
        satellite:  (Satellite) - satellite definition before the TOF is applied
        dt:         specified time-of-flight
    Outputs:
        satellite:  (Satellite) - satellite definition after the TOF is applied

    Notes:
    Uses Euler's method to calculate nu, which assumes Keplerian orbits.
    """

    ecc = satellite.orbit.ecc
    period = satellite.orbit.period

    M = 2*math.pi*dt.to(u.s)/period.to(u.s) * u.rad
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
    
    E = (E1.value % (2*math.pi))
    # nu = 2*math.atan(math.tan(E.value/2)/math.sqrt((1-ecc)/(1+E.value)))
    nu = 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(E / 2), np.sqrt(1 - ecc) * np.cos(E / 2))
    nu = (nu.value % (2*math.pi)) * u.rad

    satellite_new_orbit = Orbit.from_classical(Earth, satellite.orbit.a, satellite.orbit.ecc, satellite.orbit.inc, satellite.orbit.raan, satellite.orbit.argp, satellite.orbit.nu + nu, epoch=satellite.orbit.epoch + dt)
    satellite.orbit = satellite_new_orbit

    return satellite

def true_anomaly_to_time(satellite, nu): # updated to Review 5 prototype 

    """
    Calculates a time-of-flight for a satellite to reach a given true anomaly

    Inputs:
        satellite:  (Satellite) - current satellite definition
        nu:         target true anomaly
    Outputs:
        dt: time-of-flight
    """

    ecc = satellite.orbit.ecc
    period = satellite.orbit.period
    nu_i = satellite.orbit.nu.value

    # E1 = math.acos((ecc+math.cos(nu_i))/(1+ecc*math.cos(nu_i)))
    # M1 = E1 - ecc * math.sin(E1)

    # E2 = math.acos((ecc+math.cos(nu))/(1+ecc*math.cos(nu)))
    # M2 = E2 - ecc * math.sin(E2)

    E1 = 2 * np.arctan2(np.sqrt(1 - ecc) * np.sin(nu_i / 2), np.sqrt(1 + ecc) * np.cos(nu_i / 2))
    E2 = 2 * np.arctan2(np.sqrt(1 - ecc) * np.sin(nu / 2), np.sqrt(1 + ecc) * np.cos(nu / 2))

    M1 = E1.value - ecc * np.sin(E1)
    M2 = E2.value - ecc * np.sin(E2)
    
    dM = M2 - M1
    
    k = 0
    dt = -1
    dts = []
    while dt < 0:
        dt = (period/2/math.pi)*(2*k*math.pi + dM)
        dts.append(abs(dt))
        k = k + 1
    
    stationary_satellite_location = check_ahead_or_behind(nu_i, nu)
    dt = max(dts) if stationary_satellite_location == "ahead" else min(dts)

    return dt

def check_ahead_or_behind(nu_stationary, nu_mobile):

    """
    Provides positional correction to true_anomaly_over_time()

    Inputs:
        nu_stationary:  true anomaly of the non-maneuvering satellite
        nu_mobile:      true anomaly of the MRRV
    Outputs:
        stationary_satellite_location:  relative position along the orbit of the stationary satellite
                                        when compared to the MRRV
    """

    delta_nu = (nu_mobile - nu_stationary + math.pi) % (2 * math.pi) - math.pi
    stationary_satellite_location = "behind" if delta_nu > 0 else "ahead"

    return stationary_satellite_location

def calculate_phasing_orbit(mobile_sat, stationary_sat, dV_constraint):

    """
    Calculates the fastest phasing flight for a MRRV, given some delta V constraint

    Inputs:
        mobile_sat:     (Satellite) - MRRV satellite definition at the start of phasing flight
        stationary_sat: (Satellite) - non-maneuvering satellite definition at the start of phasing flight
        dV_constraint:  delta V constraint for phasing maneuvers

    Outputs:
        optimal_phasing_burns:  (list -> Maneuver) - details impulsive phasing burns
    """


    mu = Earth.k.to(u.km**3 / u.s**2)
    period = mobile_sat.orbit.period
    r = mobile_sat.orbit.r
    v = mobile_sat.orbit.v

    r_mag = math.sqrt(sum(component.value**2 for component in r)) * u.km
    v_mag = math.sqrt(sum(component.value**2 for component in v)) * u.km / u.s

    # Calculate if catching up or slowing down would be faster
    fallback_time = true_anomaly_to_time(stationary_sat, mobile_sat.orbit.nu.value)
    catchup_time = mobile_sat.orbit.period - fallback_time

    if fallback_time < catchup_time:
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
        
    else:
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

    optimal_v = [(component.value / v_mag.value) * v_phasing_mag.value for component in v] * u.km / u.s
    optimal_dV = optimal_v - v
    optimal_phasing_burn = Maneuver.impulse(optimal_dV)
    optimal_rendezvous_burn = Maneuver.impulse(-1 * optimal_dV)
    optimal_rendezvous_burn._dts[0] = period_phasing*n_orbits

    optimal_phasing_burns = [optimal_phasing_burn, optimal_rendezvous_burn]

    return optimal_phasing_burns

def f(t0, state, k):

    """
    Defines the acceleration under J2 perturbation

    DO NOT EDIT
    """

    du_kep = func_twobody(t0, state, k)
    ax, ay, az = J2_perturbation(
        t0, state, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad

def rocketEQ(mi, mf, I_sp):

    """
    Calculates the delta V using the rocket equation

    Inputs:
        mi:     initial satellite mass
        mf:     final satellite mass
        I_sp:    satellite I_sp
    Outputs:
        deltaV: total satellite delta V
    """

    global g0
    
    deltaV = I_sp * g0 * math.log(mi / mf)

    return deltaV

def rocketEQbackwards(deltaV, I_sp, mass, is_initial):

    """
    Calculates the amount of fuel burned for a given delta V

    Inputs:
        deltaV:     delta V of the burn
        I_sp:        satellite I_sp
        mass:       satellite mass
        is_initial: (bool) - flag for if the mass listed is the initial or final mass
    Outputs:
        fuel_burned:    mass of fuel burned
    """

    global g0

    mass_ratio = math.exp(deltaV / (I_sp * g0))
    if is_initial:
        initial_mass = mass
        final_mass = mass / mass_ratio
    else:
        initial_mass = mass / mass_ratio
        final_mass = mass

    fuel_burned = initial_mass - final_mass

    return fuel_burned

def calculate_satellite_deltaVs(MRRV, request):

    """
    Calculates delta V capabilities of the MRRV when a CS request is instantiated

    Inputs:
        MRRV:       (Satellite) - MRRV at the time the request is made
        request:    (Request)   - CS request details
    Outputs:
        total_deltaV_possible       - total delta V capability of the MRRV
        deltaV_constraint_initial   - delta V constraint for the servicing flight
        deltaV_constraint_final     - delta V constraint for the return flight
    """
    request["refuel request amount"] = min([request["refuel request amount"], MRRV.mass_prop * 0.8])
    initial_wet_mass = MRRV.mass_dry + MRRV.mass_prop
    burnable_fuel = MRRV.mass_prop - request["refuel request amount"]
    if burnable_fuel <= 0:
        print(f"Sorry, {MRRV.name} is unable to travel to the target satellite. No remaining fuel left.")
        return
    total_deltaV_possible = rocketEQ(initial_wet_mass, initial_wet_mass - burnable_fuel, MRRV.I_sp)
    # print(f"Total delta V possible: {total_deltaV_possible} m/s")

    deltaV_constraint_initial = total_deltaV_possible * request["Percent dV allocated to initial flight"]
    deltaV_constraint_final = total_deltaV_possible - deltaV_constraint_initial
    # print(f"delta V allocated to initial flight: {deltaV_constraint_initial} m/s")
    # print(f"delta V allocated to final flight:   {deltaV_constraint_final} m/s")
    
    fuel_burned_initial = rocketEQbackwards(deltaV_constraint_initial, MRRV.I_sp, initial_wet_mass, is_initial=True)
    fuel_after_initial_flight = MRRV.mass_prop - fuel_burned_initial
    fuel_after_refuel = fuel_after_initial_flight - request["refuel request amount"]

    mass_after_refuel = MRRV.mass_dry + fuel_after_refuel
    deltaV_possible_final = rocketEQ(mass_after_refuel, MRRV.mass_dry, MRRV.I_sp)
    deltaV_return_margin = deltaV_possible_final - deltaV_constraint_final

    # print(f"Actual delta V possible for final flight: {deltaV_possible_final} m/s")
    # print(f"delta V margin: {deltaV_return_margin} m/s")

    return total_deltaV_possible, deltaV_constraint_initial, deltaV_constraint_final

def calculate_operational_burn(satellite, random_seed):
    satellite_fuel_capacity = satellite.mass_prop_max
    satellite_fuel_distribution = [0 * u.kg,satellite_fuel_capacity*0.05,satellite_fuel_capacity*0.1,satellite_fuel_capacity*0.15,satellite_fuel_capacity*0.2]
    random.seed(random_seed)
    return random.sample(satellite_fuel_distribution,1)

def build_doe_dataframe(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    doe_file_path = os.path.join(script_dir, "DoE_data", filename)
    if os.path.isfile(doe_file_path):
        return pd.read_csv(doe_file_path)
    else:
        print(f"Failed to find \"{filename}\" in /data folder. Did not read csv file.")
        return

def extract_metrics(total_simulation_duration, clients, depots, mrrvs):

    LCS_operational_availability = []
    SCS_operational_availability = []

    LCS_total_prop_received = []
    LCS_total_down_time = []
    LCS_total_maintenance_time = []
    LCS_number_of_refuels = []
    LCS_longest_down_time_period = []
    LCS_longest_maintenance_period = []

    SCS_total_prop_received = []
    SCS_total_down_time = []
    SCS_total_maintenance_time = []
    SCS_number_of_refuels = []
    SCS_longest_down_time_period = []
    SCS_longest_maintenance_period = []

    mrrv_total_prop_received = 0

    for _, client in clients.items():
        if client.type == "Big Client Satellite":
            LCS_total_prop_received.append(client.total_prop_received.value)
            LCS_total_down_time.append(client.total_down_time.value)
            LCS_total_maintenance_time.append(client.time_in_maintenance.value)
            LCS_number_of_refuels.append(client.number_of_refuels)
            LCS_longest_down_time_period.append(client.longest_down_time_period.value)
            LCS_longest_maintenance_period.append(client.longest_maintenance_period.value)
            LCS_operational_availability.append((total_simulation_duration - client.time_in_maintenance.value) / total_simulation_duration)
        elif client.type == "Small Client Satellite":
            SCS_total_prop_received.append(client.total_prop_received.value)
            SCS_total_down_time.append(client.total_down_time.value)
            SCS_total_maintenance_time.append(client.time_in_maintenance.value)
            SCS_number_of_refuels.append(client.number_of_refuels)
            SCS_longest_down_time_period.append(client.longest_down_time_period.value)
            SCS_longest_maintenance_period.append(client.longest_maintenance_period.value)
            SCS_operational_availability.append((total_simulation_duration - client.time_in_maintenance.value) / total_simulation_duration)
    for _, mrrv in mrrvs.items():
        mrrv_total_prop_received += mrrv.total_prop_received.value

    # useless metrics
    LCS_metrics = [LCS_total_prop_received, LCS_total_down_time, LCS_total_maintenance_time, LCS_number_of_refuels, LCS_longest_down_time_period, LCS_longest_maintenance_period]
    LCS_metrics = sum(LCS_metrics, [])
    SCS_metrics = [SCS_total_prop_received, SCS_total_down_time, SCS_total_maintenance_time, SCS_number_of_refuels, SCS_longest_down_time_period, SCS_longest_maintenance_period]
    SCS_metrics = sum(SCS_metrics, [])

    useless_metrics = sum([LCS_metrics, SCS_metrics], [])

    # useful metrics
    LCS_percent_down_time = np.mean(LCS_total_down_time) / total_simulation_duration
    SCS_percent_down_time = np.mean(SCS_total_down_time) / total_simulation_duration
    LCS_avg_operational_availability = np.mean(LCS_operational_availability)
    SCS_avg_operational_availability = np.mean(SCS_operational_availability)
    LCS_total_prop_received = sum(LCS_total_prop_received)
    SCS_total_prop_received = sum(SCS_total_prop_received)

    useful_metrics = [LCS_percent_down_time, LCS_avg_operational_availability, LCS_total_prop_received, SCS_percent_down_time, SCS_avg_operational_availability, SCS_total_prop_received, mrrv_total_prop_received]
    

    return useless_metrics, useful_metrics
   
def write_results_to_csv(data, doe_filename, append="results", fail=False, print_status_updates=False, number_of_large_cs = 5, number_of_small_cs = 12):
    
    doe_filename = doe_filename.split(".")
    results_filename = doe_filename[0] + "_" + append + "." + doe_filename[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    doe_file_path = os.path.join(script_dir, "DoE_data", results_filename)

    new_individual_results_file = False
    new_results_file = False

    if append == "results":
        if print_status_updates:
            print('Writing results CSV...')
        if not os.path.isfile(doe_file_path):
            new_results_file = True
    elif append == "individual_results":
        if print_status_updates:
            print('Writing individual results CSV...')
        if not os.path.isfile(doe_file_path):
            new_individual_results_file = True

    with open(doe_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if new_results_file:
            writer.writerow(['case_num', 'refuel_req_threshhold', 'mrrv_prop_capacity', 'num_depots', 'num_mrrvs_per_depot', 'client_manoeuvre_regularity', 'LCS_percent_down_time', 'LCS_avg_operational_availability', 'LCS_total_prop_received', 'SCS_percent_down_time', 'SCS_avg_operational_availability', 'SCS_total_prop_received', 'mrrv_total_prop_received'])
        
        if new_individual_results_file:
            individual_results_header = ['case_num', 'refuel_req_threshhold', 'mrrv_prop_capacity', 'num_depots', 'num_mrrvs_per_depot', 'client_manoeuvre_regularity']
            
            individual_results_header.extend([f"LCS{i+1}_total_prop_received" for i in range(number_of_large_cs)])
            individual_results_header.extend([f"LCS{i+1}_total_down_time" for i in range(number_of_large_cs)])
            individual_results_header.extend([f"LCS{i+1}_total_maintenance_time" for i in range(number_of_large_cs)])
            individual_results_header.extend([f"LCS{i+1}_number_of_refuels" for i in range(number_of_large_cs)])
            individual_results_header.extend([f"LCS{i+1}_longest_down_time_period" for i in range(number_of_large_cs)])
            individual_results_header.extend([f"LCS{i+1}_longest_maintenance_period" for i in range(number_of_large_cs)])

            individual_results_header.extend([f"SCS{i+1}_total_prop_received" for i in range(number_of_small_cs)])
            individual_results_header.extend([f"SCS{i+1}_total_down_time" for i in range(number_of_small_cs)])
            individual_results_header.extend([f"SCS{i+1}_total_maintenance_time" for i in range(number_of_small_cs)])
            individual_results_header.extend([f"SCS{i+1}_number_of_refuels" for i in range(number_of_small_cs)])
            individual_results_header.extend([f"SCS{i+1}_longest_down_time_period" for i in range(number_of_small_cs)])
            individual_results_header.extend([f"SCS{i+1}_longest_maintenance_period" for i in range(number_of_small_cs)])
            
            writer.writerow(individual_results_header)

            # for i in range(number_of_large_cs):
            #     individual_results_header.append(f"LCS{i+1}_total_prop_received")
            #     individual_results_header.append(f"LCS{i+1}_total_down_time")
            #     individual_results_header.append(f"LCS{i+1}_total_maintenance_time")
            #     individual_results_header.append(f"LCS{i+1}_number_of_refuels")
            #     individual_results_header.append(f"LCS{i+1}_longest_down_time_period")
            #     individual_results_header.append(f"LCS{i+1}_longest_maintenance_period")

            # for i in range(number_of_small_cs):
            #     individual_results_header.append(f"SCS{i+1}_total_prop_received")
            #     individual_results_header.append(f"SCS{i+1}_total_down_time")
            #     individual_results_header.append(f"SCS{i+1}_total_maintenance_time")
            #     individual_results_header.append(f"SCS{i+1}_number_of_refuels")
            #     individual_results_header.append(f"SCS{i+1}_longest_down_time_period")
            #     individual_results_header.append(f"SCS{i+1}_longest_maintenance_period")
        
        if fail:
            writer.writerow(['Failed case.'])
            if print_status_updates:
                print('Writing complete.')
            return
        
        writer.writerow(data)
        if print_status_updates:
            print('Writing complete.')

    return