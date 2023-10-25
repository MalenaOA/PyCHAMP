r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on Sep 6, 2023
"""
import numpy as np
import mesa

class Well(mesa.Agent):
    """
    A class to simulate a well in an agricultural system.

    Attributes
    ----------
    unique_id : str or int
        Unique identifier for the well.
    r : float
        Radius of the well in meters.
    k : float
        Hydraulic conductivity of the aquifer in m/d.
    st : float
        Saturated thickness with respect to the well depth in meters.
    sy : float
        Specific yield of the aquifer.
    l_wt : float
        Water table lift in meters.
    eff_pump : float, optional
        Pumping efficiency. Default is 0.77.
    eff_well : float, optional
        Well efficiency. Default is 0.5.
    aquifer_id : str or int, optional
        Unique identifier for the associated aquifer.
    kwargs : dict, optional
        Additional optional arguments.
    rho : float
        Density of water in kg/m^3.
    g : float
        Acceleration due to gravity in m/s^2.
    t : int
        Current time step.
    e : float
        Energy consumption in PJ.
    """

    def __init__(self, unique_id, mesa_model, config, r, k, sy, 
                 ini_st,  ini_l_wt, ini_pumping_days=90,
                 eff_pump=0.77, eff_well=0.5, aquifer_id=None, **kwargs):
        """
        Initialize a Well object.

        Parameters
        ----------
        unique_id : str or int
            Unique identifier for the well.
        mesa_model : object
            Reference to the overarching MESA model instance.
        config : dict
            General configuration information for the model.
        r : float
            Radius of the well in meters.
        k : float
            Hydraulic conductivity of the aquifer in m/d.
        ini_st : float
            Initial saturated thickness with respect to the well depth in meters.
        sy : float
            Specific yield of the aquifer.
        ini_l_wt : float
            Initial water table lift in meters.
        eff_pump : float, optional
            Pumping efficiency. Default is 0.77.
        eff_well : float, optional
            Well efficiency. Default is 0.5.
        aquifer_id : str or int, optional
            Unique identifier for the associated aquifer.
        kwargs : dict, optional
            Additional optional arguments.
        """
        # MESA required attributes => (unique_id, model)
        super().__init__(unique_id, mesa_model)
        self.agt_type = "Well"

        self.unique_id, self.r, self.k, self.st, self.sy, self.l_wt = \
            unique_id, r, k, ini_st, sy, ini_l_wt
        self.eff_pump, self.eff_well = eff_pump, eff_well
        self.aquifer_id = aquifer_id
        self.load_config(config)

        # Load other kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.tr = self.st * k    # Transmissivity

        self.t = 0

        # Container
        self.pumping_days = ini_pumping_days
        self.withdrawal = None # m-ha

    def load_config(self, config):
        """
        Load well-related configurations from the model's general configuration.

        Parameters
        ----------
        config : dict
            General configuration information for the model.

        Returns
        -------
        None.

        """
        config_well = config["well"]
        self.rho = config_well["rho"]
        self.g = config_well["g"]

    def step(self, withdrawal, dwl, pumping_rate, l_pr, pumping_days=90):
        """
        Simulate the well for one time step based on water withdrawal and other parameters.

        Parameters
        ----------
        withdrawal : float
            The total volume of water to be withdrawn from the well in m-ha.
        dwl : float
            Change in groundwater level in meters.
        pumping_rate : float
            Pumping rate in m^3/d.
        l_pr : float
            Pressure lift in meters.

        Returns
        -------
        float
            Energy consumption for the current time step in PJ.

        Attributes Modified
        -------------------
        t : int
            Updated time step.
        l_wt : float
            Updated water table lift.
        st : float
            Updated saturated thickness.
        tr : float
            Updated transmissivity.
        e : float
            Updated energy consumption.
        """
        self.t +=1
        self.pumping_days = pumping_days
        # Update saturated thickness and water table lift based on groundwater
        # level change
        self.l_wt -= dwl
        self.st += dwl
        tr_ = self.st * self.k # Update Transmissivity
        # cannot divided by zero
        if tr_ < 0.001:
            self.tr = 0.001
        else:
            self.tr = tr_
        
        self.withdrawal = withdrawal
        l_wt = self.l_wt

        r, tr, sy = self.r, self.tr, self.sy
        eff_well, eff_pump = self.eff_well, self.eff_pump
        rho, g = self.rho, self.g

        # Calculate energy consumption
        m_ha_2_m3 = 10000
        fpitr = 4 * np.pi * tr
        ftrd = 4 * tr * pumping_days
        
        l_cd_l_wd = (1+eff_well) * pumping_rate/fpitr \
                    * (-0.5772 - np.log(r**2*sy/ftrd)) * m_ha_2_m3
        l_t = l_wt + l_cd_l_wd + l_pr
        e = rho * g * m_ha_2_m3 / eff_pump / 1e15 * withdrawal * l_t     # PJ

        # Record energy consumption
        self.e = e

        return e
    
# for i in range(20):
#     m_ha_2_m3 = 10000
#     tr = (i+1) * 80
#     fpitr = 4 * np.pi * tr
#     l_cd_l_wd = (1+0.5) * 5000/fpitr \
#                 * (-0.5772 - np.log(0.4064**2*0.05/fpitr)) * m_ha_2_m3    
#     print(l_cd_l_wd)
    
