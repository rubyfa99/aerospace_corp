""" Dynamic Space Operations Team 21
    Title: Object Class Definition
    Purpose: Define Object Classes with relevant Space information, allows for updates to be made to each
    object for certain parameters."""
"""
Notes: 
- N ADD IN Instatiate dict.. 
- could add in such that this object updates itself everytime it is called for anything ever

- NEED TO CHANGE SUCH THAT THE ASTROPY UNITS ARE INPUTTED SO ITS EASIER TO SEE FROM USER SIDE INSTEAD OF HERE...
- Need a propagator... to tell it where it is in future time.
- Need a propagator... to tell it where it is in future time.
"""

#import Poliastro
import simpy 
import numpy as np
from astropy import units as u 
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.twobody import propagation

global r_earth, mu, g0
r_earth = 6371 * u.km # km
mu = 3.986e5 * (u.km**3/u.s**2) # km3/s2
g0 = 0.00981 * u.km/(u.s**2)  # km/s2


class Satellite:
    """The Class defines any generic Satellite object with the respective necessary metrics, cartesian parameters need to be specified here for operation. """
    def __init__(self, env, name, mass_dry, mass_prop, mass_prop_max, I_sp, refuel_rate_in, refuel_rate_out, initial_coe=None, initial_rv=None, color=None): #position, velocity, semi_major_axis, eccentricity, inclination, RAAN, omega, true_anomaly):
        self.env = env  # SimPy environment
        self.name = name       
        self.current_time=env.now * u.s

        # Satellite Characteristics
        self.mass_dry = mass_dry 
        self.mass_prop = mass_prop
        self.mass_prop_max = mass_prop_max
        self.I_sp = I_sp 
        self.refuel_rate_in = refuel_rate_in 
        self.refuel_rate_out = refuel_rate_out 
        
        # Graphic specific parameters
        self.color = color

        self.history = []  # Track state history with timestamps
        # self.event_queue = []  # Queue for scheduled events
        self.action_count = 0  # Initialize action counter
        self.state = "Standby"

        # Initializing the Orbit
        self.orb = None

        # If the user specifies position and velocity
        if initial_rv is not None:
            position = initial_rv["position"] # Cartesian (x, y, z) in km
            velocity = initial_rv["velocity"] # Cartesian (vx, vy, vz) in km/s

            # Generate orbit from r, v vectors
            self.orb = Orbit.from_vectors(Earth, r = position, v = velocity, epoch=self.current_time)
        # If the user specifies the 6 orbital parameters 
        elif initial_coe["semi_major_axis"] is not None and initial_coe["eccentricity"] is not None and initial_coe["inclination"] is not None and initial_coe["RAAN"] is not None and initial_coe["omega"] is not None and initial_coe["true_anomaly"] is not None:
            semi_major_axis = initial_coe["semi_major_axis"] 
            eccentricity = initial_coe["eccentricity"] 
            inclination = initial_coe["inclination"] 
            RAAN = initial_coe["RAAN"] 
            omega = initial_coe["omega"] 
            true_anomaly = initial_coe["true_anomaly"]

            # Generate orbit from COE
            self.orb = Orbit.from_classical(Earth, a = semi_major_axis, ecc = eccentricity, inc = inclination, raan = RAAN, argp = omega, nu = true_anomaly, epoch=self.current_time)
        else: 
            raise ValueError("Sorry, you need to make sure to have all cartesian or all orbital parameters included.")


    def log_state(self):
        """Log the current state of the satellite with timestamp."""
        self.action_count += 1
        log_entry = {
            "timestamp": f" {self.env.now} s",
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


    def maneuver(self, delta_V, execution_time=None): #FOR Dealing w current use case... could delete
        """
        Schedule or execute a maneuver and return a SimPy event.
        """
        if execution_time is not None and execution_time > self.env.now:
            # Schedule for later
            event = self.env.process(self._maneuver(delta_V, execution_time))
            return event
        else:
            # Execute immediately
            event = self.env.process(self._maneuver(delta_V, self.env.now))
            return event
    
    def _maneuver(self, dVarr, execution_time):
        """
        Internal maneuver process. Excecute maneuver, return burn, update sc mass.
        """
        print(f"Starting maneuver at time {self.env.now}. Execution time: {execution_time}")
        yield self.env.timeout(execution_time - self.env.now)
        dVarr = np.array(dVarr) * u.km / u.s if not isinstance(dVarr, u.Quantity) else dVarr 
        burn = Maneuver.impulse(dVarr)
        finalOrbit = self.orbit.apply_maneuver(burn)

        # Updating the object's orbit
        self.orbit = finalOrbit 
        
        # Obtaining dV and time expended
        dV_expended = burn.get_total_cost()
        time_expended = burn.get_total_time()
        
        # Calculate the final mass after the burn
        m_final = (self.mass_prop+self.mass_dry) * np.exp(-dV_expended / (self.I_sp * g0))
        self.mass_prop = m_final - self.mass_dry #Update the propellant mass attribute
        
        self.log_state()
        return burn, time_expended, dV_expended

    def update_time_state(self): #RECONFIGURE TO DANIEL'S CODE!
        """
        Retrieve all time-dependent information of the satellite at a given time
        as a dictionary. Defaults to the current simulation time if no time
        is provided. This is a second version of above. 
        """

        # SEE IF MANEUVER HAS BEEN PERFORMED IN PAST.. 
        
        # Calculate mean anomaly
        T = 2 * np.pi * np.sqrt((self.semi_major_axis)**3 / mu)  # Orbital period
        mean_motion = 2 * np.pi / T  # Mean motion (rad/s)
        self.current_time = self.env.now * u.s #DEBUGGING --> SAW THAT CURRENT TIME WAS STILL 0 & NOT UPDATING WITH SIM TIME OF =400s 
        M = (mean_motion * self.current_time) % (2 * np.pi)  # Mean anomaly

        # Solve Kepler's equation for eccentric anomaly
        E = M  # Initial guess
        for _ in range(100):  # Newton-Raphson iteration
            E_new = E - (E - self.orb.ecc * np.sin(E * u.rad) - M) / (1 - self.orb.ecc * np.cos(E * u.rad))
            if abs(E_new - E) < 1e-6:
                E = E_new
                break
            E = E_new

        # Calculate true anomaly
        nu = 2 * np.arctan2(
            np.sqrt(1 + self.orb.ecc) * np.sin(E*u.rad / 2),
            np.sqrt(1 - self.orb.ecc) * np.cos(E*u.rad / 2)
        )

        # Calculate position and velocity in perifocal coordinates
        p = self.orb.a * (1 - self.orb.ecc**2)
        r_pqw = (p / (1 + self.orb.ecc * np.cos(nu))) * np.array([np.cos(nu), np.sin(nu), 0])
        v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), self.orb.ecc + np.cos(nu), 0])

        # Convert to ECI coordinates
        inc, RAAN, omega = map(np.radians, [self.orb.inc, self.orb.raan, self.orb.argp])
        R3_RAAN = np.array([
            [np.cos(-RAAN), np.sin(-RAAN), 0],
            [-np.sin(-RAAN), np.cos(-RAAN), 0],
            [0, 0, 1]
        ])

        R1_i = np.array([
            [1, 0, 0],
            [0, np.cos(-inc), np.sin(-inc)],
            [0, -np.sin(-inc), np.cos(-inc)]
        ])

        R3_omega = np.array([
            [np.cos(-omega), np.sin(-omega), 0],
            [-np.sin(-omega), np.cos(-omega), 0],
            [0, 0, 1]
        ])

        # Total rotation matrix
        rotation_matrix = R3_RAAN @ R1_i @ R3_omega

        # Transform position and velocity to ECI
        r_eci = rotation_matrix @ r_pqw
        v_eci = rotation_matrix @ v_pqw
      
        #Generate new orbit 
        self.orbit = Orbit.from_vectors(Earth, r = r_eci, v = v_eci)
         
    def get_state_at_time(self, time=None):
        """
        Retrieve all time-dependent information of the satellite at a given time
        as a dictionary. Defaults to the current simulation time if no time
        is provided. This is a propagator. 
        """
        # Default to current simulation time if no time is provided
        if time is None:
            time = self.env.now

        #   PLACEHOLDER FOR IF MANEUVERS ARE SCHEDULED 
        #if man_scheduled is True 
        #   check time of data request, see if falls before/after specific maneuver, then "apply that maneuver",
        #       repropagate data, regenerate position & velocity and true anomaly 
        #
        rVec, Vvec = propagation.farnocchia(Earth, self.orb.r, self.orb.v, time)

        # Construct the data dictionary
        data = {
            "Satellite Name": self.name,
            "Dry Mass (kg)": self.mass_dry,
            "Propellant Mass (kg)": self.mass_prop,
            "Maximum Propellant Capacity (kg)": self.mass_prop_max,
            "Specific Impulse (s)": self.I_sp,
            "Refuel Rate (kg/s)": {
                "In": self.refuel_rate_in,
                "Out": self.refuel_rate_out
            },
            "Position (km)": rVec,
            "Velocity (km/s)": Vvec,
            "Simulation Timestamp (s)": time
        }

        return data 

'''The sub-classes of Satellite, inheriting all characteristics of the Satellite.'''

class Client(Satellite):
    def __init__(self, env, name, mass_dry, mass_prop, mass_prop_max, I_sp, refuel_rate_in, refuel_rate_out, initial_coe=None, initial_rv=None, color=None):
        super().__init__(
            env=env,
            name=name,
            mass_dry=mass_dry,
            mass_prop=mass_prop,
            mass_prop_max=mass_prop_max,
            I_sp=I_sp,
            refuel_rate_in=refuel_rate_in,
            refuel_rate_out=refuel_rate_out,
            initial_coe=initial_coe,
            initial_rv=initial_rv,
            color=color
        )
        self.type = "client"


class MRRV(Satellite):
    def __init__(self, env, name, mass_dry, mass_prop, mass_prop_max, I_sp, refuel_rate_in, refuel_rate_out, initial_coe=None, initial_rv=None, color=None, deorbit_dV_needed=None):
        super().__init__(
            env=env,
            name=name,
            mass_dry=mass_dry,
            mass_prop=mass_prop,
            mass_prop_max=mass_prop_max,
            I_sp=I_sp,
            refuel_rate_in=refuel_rate_in,
            refuel_rate_out=refuel_rate_out,
            initial_coe=initial_coe,
            initial_rv=initial_rv,
            color=color
        )
        self.deorbit_dV_needed = deorbit_dV_needed
        self.type = "mrrv"

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
    def __init__(self, env, name, mass_dry, mass_prop, mass_prop_max, I_sp, refuel_rate_in, refuel_rate_out, initial_coe=None, initial_rv=None, color=None):
        super().__init__(
            env=env,
            name=name,
            mass_dry=mass_dry,
            mass_prop=mass_prop,
            mass_prop_max=mass_prop_max,
            I_sp=I_sp,
            refuel_rate_in=refuel_rate_in,
            refuel_rate_out=refuel_rate_out,
            initial_coe=initial_coe,
            initial_rv=initial_rv,
            color=color
        )        
        # Additional attributes specific to Depot satellites can be added here
        self.type = "depot"




