""" Dynamic Space Operations Team 21
    Title: Object Class Definition
    Purpose: Define Object Classes with relevant Space information, allows for updates to be made to each
    object for certain parameters."""
"""
Notes: 
- added in option to auto populate mass distributions and dry mass if a fuel type and amount is specified
"""

#import Poliastro
import numpy as np
from astropy import units as u 
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.twobody import propagation
from helper_functions import time_to_true_anomaly
from poliastro.constants import J2000

global r_earth, mu, g0
r_earth = 6371 * u.km # km
mu = 3.986e5 * (u.km**3/u.s**2) # km3/s2
g0 = 0.00981 * u.km/(u.s**2)  # km/s2
start_epoch = 0 * u.s #or J2000

class Satellite:
    """The Class defines any generic Satellite object with the respective necessary metrics, cartesian parameters need to be specified here for operation. """
    def __init__(self, name, mass_dry=None, mass_prop=0, mass_prop_max=0,
             I_sp=None, fuel_transfer_rate_in=0, fuel_transfer_rate_out=0,
             initial_coe=None, initial_rv=None, color=None, fuel_type=None):
        self.name = name       

        # Satellite Characteristics
        self.mass_prop = mass_prop
        self.mass_prop_max = mass_prop_max
        self.fuel_transfer_rate_in = fuel_transfer_rate_in 
        self.fuel_transfer_rate_out = fuel_transfer_rate_out 
        
        if fuel_type:
            fuel_data = PROPELLANT_PROPERTIES.get(fuel_type.lower())
            if not fuel_data:
                raise ValueError(f"Fuel type '{fuel_type}' not recognized.")
            self.fuel_type = fuel_type
            self.I_sp = (fuel_data["Isp"] if I_sp is None else I_sp) * u.s

            # Calculate tank volume and tank mass
            tank_volume = mass_prop / fuel_data["density"]  # in m³
            tank_mass = tank_volume * fuel_data["tank_mass_per_m3"]

            # Estimate dry mass if not provided
            if mass_dry is None:
                tank_to_dry_ratio = 0.07  # 7% of dry mass is tank
                self.mass_dry = (tank_mass / tank_to_dry_ratio) * u.kg
            else:
                self.mass_dry = mass_dry
        else:
            self.mass_dry = mass_dry
            self.I_sp = I_sp

        # Graphic specific parameters
        self.color = color

        self.history = []  # Track state history with timestamps
        # self.event_queue = []  # Queue for scheduled events
        self.action_count = 0  # Initialize action counter
        self.state = "Available"

        # Initializing the Orbit
        self.orbit = None

        # If the user specifies position and velocity
        if initial_rv is not None:
            position = initial_rv["position"] # Cartesian (x, y, z) in km
            velocity = initial_rv["velocity"] # Cartesian (vx, vy, vz) in km/s

            # Generate orbit from r, v vectors
            self.orbit = Orbit.from_vectors(Earth, r = position, v = velocity, epoch=self.current_time)
        # If the user specifies the 6 orbital parameters 
        elif initial_coe["semi_major_axis"] is not None and initial_coe["eccentricity"] is not None and initial_coe["inclination"] is not None and initial_coe["RAAN"] is not None and initial_coe["omega"] is not None and initial_coe["true_anomaly"] is not None:
            semi_major_axis = initial_coe["semi_major_axis"] 
            eccentricity = initial_coe["eccentricity"] 
            inclination = initial_coe["inclination"] 
            RAAN = initial_coe["RAAN"] 
            omega = initial_coe["omega"] 
            true_anomaly = initial_coe["true_anomaly"]

            # Generate orbit from COE
            self.orbit = Orbit.from_classical(Earth, a = semi_major_axis, ecc = eccentricity, inc = inclination, raan = RAAN, argp = omega, nu = true_anomaly, epoch=start_epoch)
        else: 
            raise ValueError("Sorry, you need to make sure to have all cartesian or all orbital parameters included.")
    
    def get_mass_distribution(self):
        """Returns a mass breakdown by subsystem as a dict."""

        # Adjusted to total 100% (no payload)
        dist = {
            "Structure & Mechanisms": 0.24,
            "Thermal Control": 0.04,
            "Power": 0.17,
            "TT&C": 0.04,
            "On-Board Processing": 0.03,
            "ADCS": 0.06,
            "Propulsion (tank etc.)": 0.39, # 0.07 + 0.32 tank + payload mass, assumption
            "Other": 0.03
        }

        breakdown = {}
        for key, frac in dist.items():
            breakdown[key] = round(self.mass_dry * frac, 2)
        return breakdown


    def log_state(self):
        """Log the current state of the satellite with timestamp."""
        self.action_count += 1
        log_entry = {
            #"timestamp": f" {} s",
            "action": f"Action #{self.action_count}",
            "position": self.orb.r.value.tolist() if self.position is not None else None,
            "velocity": self.orb.v.value.tolist() if self.velocity is not None else None,
            "semi_major_axis": self.orb.a,
            "eccentricity": self.orb.ecc,
            "inclination": self.orb.inc,
            "RAAN": self.orb.raan,
            "omega": self.orb.argp,
            "true_anomaly": self.orb.nu,
            "propellant_mass": self.mass_prop
        }
        self.history.append(log_entry)


    def print_history(self):
        """
        Funtion to print and display history of object's actions. Preliminary and 
        for debugging. Higher fidelity = export timestamps and actions to .csv 
        """
        for entry in self.history:
            print(f"\n--- Action Log ---")
            for key, value in entry.items():
                print(f"{key}: {value}")

'''The sub-classes of Satellite, inheriting all characteristics of the Satellite.'''

class Client(Satellite):
    def __init__(self, name, mass_dry, mass_prop, mass_prop_max, I_sp, fuel_transfer_rate_in, fuel_transfer_rate_out, initial_coe=None, initial_rv=None, color=None, fuel_depletion_rate=None, maneuver_regularity=None, type=None, rng=None):
        super().__init__(
            name=name,
            mass_dry=mass_dry,
            mass_prop=mass_prop,
            mass_prop_max=mass_prop_max,
            I_sp=I_sp,
            fuel_transfer_rate_in=fuel_transfer_rate_in,
            fuel_transfer_rate_out=fuel_transfer_rate_out,
            initial_coe=initial_coe,
            initial_rv=initial_rv,
            color=color
        )
        self.type = type
        self.down_start_time = None  # Timestamp when the satellite enters the "down" state
        self.total_down_time = 0 * u.s  # Total time spent in the "down" state
        self.fuel_depletion_rate = fuel_depletion_rate
        self.maneuver_regularity = maneuver_regularity # 1-100, 1 = frequent, but small burns, 100 = infrequent, but large burns
        self.number_of_refuels = 0
        self.total_prop_received = 0 * u.kg
        self.time_in_maintenance = 0 * u.s
        self.longest_maintenance_period = 0 * u.s
        self.longest_down_time_period = 0 * u.s
        self.time_last_burn = 0
        self.rng = rng
        self.burn_this_timestep = 0 * u.kg


class MRRV(Satellite):
    def __init__(self, name, mass_dry, mass_prop, mass_prop_max, I_sp, fuel_transfer_rate_in, fuel_transfer_rate_out, initial_coe=None, initial_rv=None, color=None, deorbit_dV_needed=None, home_depot=None):
        super().__init__(
            name=name,
            mass_dry=mass_dry,
            mass_prop=mass_prop,
            mass_prop_max=mass_prop_max,
            I_sp=I_sp,
            fuel_transfer_rate_in=fuel_transfer_rate_in,
            fuel_transfer_rate_out=fuel_transfer_rate_out,
            initial_coe=initial_coe,
            initial_rv=initial_rv,
            color=color
        )
        self.deorbit_dV_needed = deorbit_dV_needed
        self.type = "mrrv"
        self.home_depot = home_depot
        self.number_of_refuels = 0
        self.total_prop_received = 0 * u.kg

    def update_deorbit(self, new_deorbit_dV_needed):
        self.deorbit_dV_needed = new_deorbit_dV_needed
        self.log_state()

    # def log_state(self):
    #     # Call the parent class's log_state to log general information
    #     super().log_state()

    #     # Append additional information specific to MRRV
    #     if self.deorbit_dV_needed is not None:
    #         self.history[-1] += f"Deorbit dV Needed: {self.deorbit_dV_needed}\n"


class Depot(Satellite):
    def __init__(self, name, mass_dry, mass_prop, mass_prop_max, I_sp, fuel_transfer_rate_in, fuel_transfer_rate_out, initial_coe=None, initial_rv=None, color=None, current_no_mrrvs=None, max_parking_spaces=None):
        super().__init__(
            name=name,
            mass_dry=mass_dry,
            mass_prop=mass_prop,
            mass_prop_max=mass_prop_max,
            I_sp=I_sp,
            fuel_transfer_rate_in=fuel_transfer_rate_in,
            fuel_transfer_rate_out=fuel_transfer_rate_out,
            initial_coe=initial_coe,
            initial_rv=initial_rv,
            color=color
        )        
        # Additional attributes specific to Depot satellites can be added here
        self.type = "depot"
        self.current_no_mrrvs = current_no_mrrvs
        self.max_parking_spaces = max_parking_spaces

# Define monopropellant properties
PROPELLANT_PROPERTIES = {
    "hydrazine": {"density": 1011, "Isp": 230, "tank_mass_per_m3": 40},
    "af-m315e": {"density": 1350, "Isp": 250, "tank_mass_per_m3": 30},
    "lmp-103s": {"density": 1320, "Isp": 245, "tank_mass_per_m3": 28}
}