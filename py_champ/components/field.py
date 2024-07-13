# The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
# Email: chungyi@vt.edu
# Last modified on Dec 30, 2023
import mesa
import numpy as np
import os
import pandas as pd

class Field(mesa.Agent):
    """
    This module is a field simulator.

    Parameters
    ----------
    unique_id : int
        A unique identifier for this agent.
    model
        The model instance to which this agent belongs.
    settings : dict
        A dictionary containing initial settings for the field, which include field
        area, water yield curves, technology pumping rate coefficients,
        climate data id, and initial conditions.

        - 'field_area': The total area of the field [ha].
        - 'water_yield_curves': Water yield response curves for different crops.
        - 'tech_pumping_rate_coefs': Coefficients for calculating pumping rates based on irrigation technology. Pumping rate [m-ha/day] = a * annual withdrawal [m-ha] + b
        - 'prec_aw_id': Identifier for available precipitation data.
        - 'init': Initial conditions: irrigation technology, crop type, and field type.

        >>> # A sample settings dictionary
        >>> settings = {
        >>>     "field_area": 50.,
        >>>     "water_yield_curves": {
        >>>     "corn": [ymax [bu], wmax [cm], a, b, c, min_yield_ratio]}, # Replace variables with actual values
        >>>     "tech_pumping_rate_coefs": {
        >>>         "center pivot LEPA": [a, b, l_pr [m]]}, # Replace variables with actual values
        >>>     "prec_aw_id": None,
        >>>     "init":{
        >>>         "tech": None,
        >>>         "crop": None,
        >>>         "field_type": None, # "optimize" or "irrigated" or "rainfed"
        >>>         },
        >>>    }

    **kwargs
        Additional keyword arguments that can be dynamically set as field agent attributes.

    Attributes
    ----------
    agt_type : str
        The type of the agent, set to 'Field'.
    te : str
        The current irrigation technology.
    crops : list
        The list of crops planted in the field.
    field_type : str
        The type of the field ("optimize" or "irrigated" or "rainfed").
    t : int
        The current time step, initialized to zero.
    irr_vol : float or None
        The total volume of irrigation applied [m-ha].
    yield_rate_per_field : float or None
        The averaged yield rate across the fields [bu/ha].
    irr_vol_per_field : float or None
        The averaged irrigation volume per fields [m-ha].

    Notes
    -----
    - The yield is measured in bushels [1e4 bu].
    - The irrigation volume is measured in meter-hectares [m-ha].
    - Field area should be provided in hectares [ha].
    """

    def __init__(self, unique_id, model, settings: dict, **kwargs):
        """Initialize a Field agent in the Mesa model."""
        # MESA required attributes => (unique_id, model)
        super().__init__(unique_id, model)
        self.agt_type = "Field"

        # Initialize attributes
        self.load_settings(settings)

        # Initialize and update tech
        self.te = self.init["tech"]
        self.update_irr_tech(self.init["tech"])

        # Initialize and update crop
        crop_options = self.model.crop_options
        i_crop = np.zeros((self.n_s, self.n_c, 1))
        ini_crop = self.init["crop"]
        if isinstance(ini_crop, str):
            i_c = crop_options.index(ini_crop)
            i_crop[:, i_c, 0] = 1
            self.crops = [ini_crop] * self.n_s
        else:
            self.crops = ini_crop
            for s, c in enumerate(ini_crop):
                i_c = crop_options.index(c)
                i_crop[s, i_c, 0] = 1
        self.i_crop = i_crop
        self.update_crops(i_crop)

        # Initialize field type
        self.field_type = self.init["field_type"]

        # Additional attributes from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        # [Deprecated] self.block_w_interval_for_corn = kwargs.get("block_w_interval_for_corn")

        # Initialize other variables
        self.t = 0
        self.irr_vol = None
        self.yield_rate_per_field = None  # Averaged value across a field
        self.irr_vol_per_field = None  # Averaged value across a field

    def load_settings(self, settings: dict):
        """
        Load the field settings from a dictionary.

        Parameters
        ----------
        settings : dict
            A dictionary containing settings for the field, including field
            area, water yield curves for crops, and technological coefficients.
        """
        crop_options = self.model.crop_options
        self.field_area = settings["field_area"]
        self.water_yield_curves = settings["water_yield_curves"]
        self.tech_pumping_rate_coefs = settings["tech_pumping_rate_coefs"]
        self.prec_aw_id = settings["prec_aw_id"]
        self.init = settings["init"]

        self.n_s = self.model.area_split
        self.n_c = len(crop_options)

        crop_par = np.array([self.water_yield_curves[c] for c in crop_options])
        self.ymax = crop_par[:, 0].reshape((-1, 1))  # (n_c, 1)
        self.wmax = crop_par[:, 1].reshape((-1, 1))  # (n_c, 1)
        self.a = crop_par[:, 2].reshape((-1, 1))  # (n_c, 1)
        self.b = crop_par[:, 3].reshape((-1, 1))  # (n_c, 1)
        self.c = crop_par[:, 4].reshape((-1, 1))  # (n_c, 1)
        try:
            self.min_y_ratio = crop_par[:, 5].reshape((-1, 1))  # (n_c, 1)
        except:
            self.min_y_ratio = np.zeros((self.n_c, 1))

        self.unit_area = self.field_area / self.n_s

    def update_irr_tech(self, i_te):
        """
        Update the irrigation technology used in the field based on given
        indicator array. The dimension of the array should be (n_te).

        Parameters
        ----------
        i_te : 1d array or str
            Indicator array or string representing the chosen irrigation
            technology for the next year. The dimension of the array should be
            (n_te).

        Returns
        -------
        None
        """
        if isinstance(i_te, str):
            new_te = i_te
        else:
            # Use argmax instead of "==1" to avoid float numerical issue.
            new_te = self.model.tech_options[np.argmax(i_te)]
        self.a_te, self.b_te, self.l_pr = self.tech_pumping_rate_coefs[new_te]
        self.pre_te = self.te
        self.te = new_te

    def update_crops(self, i_crop):
        """
        Update the crop types for each area split based on the given indicator
        array. The dimension of the array should be (n_s, n_c, 1).

        Parameters
        ----------
        i_crop : 3d array
            Indicator array representing the chosen crops for the next year for
            each area split. The dimension of the array should be (n_s, n_c, 1).

        Returns
        -------
        None
        """
        n_s = self.n_s
        crop_options = self.model.crop_options
        # Use argmax instead of ==1 to avoid float numerical issue
        self.pre_i_crop = self.i_crop
        crops = [crop_options[np.argmax(i_crop[s, :, 0])] for s in range(n_s)]
        self.crops = crops
        self.i_crop = i_crop

    def step(self, irr_depth, i_crop, i_te, prec_aw: dict) -> tuple:
        """
        Perform a single step of field operation, calculating yields and
        irrigation volumes.

        Parameters
        ----------
        irr_depth : 3d array
            The depth of irrigation applied [cm]. Dimensions: (n_s, n_c, 1)
        i_crop : 3d array
            Indicator array representing the chosen crops for each area split.
            Dimensions: (n_s, n_c, 1).
        i_te : 1d array or str
            Indicator array or string representing the chosen irrigation
            technology. Dimensions: (n_te).
        prec_aw : dict
            A dictionary of available precipitation for each crop.
            {"corn": 27.02, "sorghum": 22.81}

        Returns
        -------
        tuple
            A tuple containing yield [1e4 bu], average yield rate [-], and
            total irrigation volume [m-ha].

        Notes
        -----
        This method calculates the yield based on the applied irrigation, chosen crops,
        install technology, and available precipitation.
        """
        self.t += 1

        a = self.a
        b = self.b
        c = self.c
        ymax = self.ymax
        wmax = self.wmax
        n_s = self.n_s
        unit_area = self.unit_area
        crop_options = self.model.crop_options

        ### Yield calculation
        irr_depth = irr_depth.copy()[:, :, [0]]
        prec_aw_ = np.ones(irr_depth.shape)
        for ci, crop in enumerate(crop_options):
            prec_aw_[:, ci, :] = prec_aw[crop]

        w = irr_depth + prec_aw_
        w = w * i_crop
        w_ = w / wmax  # normalized applied water
        w_ = np.minimum(w_, 1)
        y_ = a * w_**2 + b * w_ + c  # normalized yield
        y_ = np.maximum(0, y_)
        y_ = y_ * i_crop

        # No force fallow here. The decisions are made. 0 yield only mean failure in that decisions.
        # Force a margin cutoff
        # min_y_ratio = np.tile(self.min_y_ratio, (self.n_s, 1, 1))
        # if "fallow" in crop_options:
        #     i_crop[y_ < min_y_ratio] = 0
        #     fallow_ind = crop_options.index("fallow")
        #     zero_columns = np.all(i_crop == 0, axis=1)
        #     # Set the fourth row of those columns to 1
        #     i_crop[zero_columns[0], fallow_ind] = 1
        # y_[y_ < min_y_ratio] = 0

        self.update_crops(i_crop)  # update pre_i_crop

        y = y_ * ymax * unit_area * 1e-4  # 1e4 bu

        cm2m = 0.01
        v_c = irr_depth * unit_area * cm2m  # m-ha
        irr_vol = np.sum(v_c)  # m-ha
        avg_y_y = np.sum(y_) / n_s
        avg_w = np.sum(w) / n_s

        ### Tech (for the pumping cost calculation in Finance module)
        self.update_irr_tech(i_te)  # update tech
        a_te = self.a_te
        b_te = self.b_te
        pumping_rate = a_te * irr_vol + b_te  # (McCarthy et al., 2020)
        self.pumping_rate = pumping_rate  # m-ha/day

        # record
        self.y = y
        self.w = avg_w
        self.yield_rate_per_field = avg_y_y
        self.irr_vol_per_field = irr_vol  # m-ha

        return y, avg_y_y, irr_vol


class Field4SingleFieldAndWell(mesa.Agent):
    """ Simulate a field agent in the model. """
    def __init__(self, unique_id, model, settings: dict, **kwargs):
        """Initialize a Field agent.

        Parameters
        ----------
        unique_id : int
            A unique identifier for this agent.
        model
            The mesa model instance to which this agent belongs.
        settings : dict
            A dictionary containing initial settings for the field, which include field
            area, water yield curves, climate data id, and initial conditions.
        """
        # MESA required attributes => (unique_id, model)
        super().__init__(unique_id, model)
        self.agt_type = "Field"

        # Initialize attributes
        self.load_settings(settings)

        # Initialize and update crop
        crop_options = self.model.crop_options
        i_crop = np.zeros((self.n_c, 1))
        ini_crop = self.init["crop"]
        self.crop = ini_crop
        i_c = crop_options.index(ini_crop)
        i_crop[i_c, 0] = 1
        self.i_crop = i_crop
        self.update_crops(i_crop)
        
        # Initialize field type
        self.field_type = self.init["field_type"]

        # Additional attributes from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Initialize other variables
        self.t = 0
        self.irr_vol = None
        self.yield_rate_per_field = None    # Averaged value across a field 
        self.irr_vol_per_field = None       # Averaged value across a field 

    def load_settings(self, settings: dict):
        """
        Load the field settings from a dictionary.
    
        Parameters
        ----------
        settings : dict
            A dictionary containing settings for the field, including field
            area, water yield curves for crops, and technological coefficients.
        """
        crop_options = self.model.crop_options
        self.field_area = settings["field_area"]
        self.water_yield_curves = settings["water_yield_curves"]
        self.prec_aw_id = settings["prec_aw_id"]
        self.init = settings["init"]
        
        self.n_c = len(crop_options)

        crop_par = np.array([self.water_yield_curves[c] for c in crop_options])
        self.ymax = crop_par[:, 0].reshape((-1, 1))     # (n_c, 1)
        self.wmax = crop_par[:, 1].reshape((-1, 1))     # (n_c, 1)
        self.a = crop_par[:, 2].reshape((-1, 1))        # (n_c, 1)
        self.b = crop_par[:, 3].reshape((-1, 1))        # (n_c, 1)
        self.c = crop_par[:, 4].reshape((-1, 1))        # (n_c, 1)
        try:
            self.min_y_ratio = crop_par[:, 5].reshape((-1, 1))    # (n_c, 1)
        except:
            self.min_y_ratio = np.zeros((self.n_c, 1))
            
        self.unit_area = self.field_area

    def update_crops(self, i_crop):
        """
        Update the crop types for each area split based on the given indicator
        array. The dimension of the array should be (n_c, 1).

        Parameters
        ----------
        i_crop : 2d array
            Indicator array representing the chosen crops for the next year. 
            The dimension of the array should be (n_c, 1).

        Returns
        -------
        None
        """
        crop_options = self.model.crop_options
        # Use argmax instead of ==1 to avoid float numerical issue
        self.pre_i_crop = self.i_crop
        #crops = [crop_options[np.argmax(i_crop[s, :, 0])] for s in range(n_s)]
        crop = crop_options[np.argmax(i_crop[:, 0])]
        self.crop = crop
        self.i_crop = i_crop

    def step(self, irr_depth, i_crop, prec_aw: dict) -> tuple:
        """
        Perform a single step of field operation, calculating yields and
        irrigation volumes.
    
        Parameters
        ----------
        irr_depth : 3d array
            The depth of irrigation applied [cm]. Dimensions: (n_s, n_c, 1)
        i_crop : 3d array
            Indicator array representing the chosen crops for each area split.
            Dimensions: (n_s, n_c, 1).
        prec_aw : dict
            A dictionary of available precipitation for each crop. 
            {"corn": 27.02, "sorghum": 22.81}
    
        Returns
        -------
        tuple
            A tuple containing yield [1e4 bu], average yield rate [-], and 
            total irrigation volume [m-ha].
    
        Notes
        -----
        This method calculates the yield based on the applied irrigation, chosen crops, 
        install technology, and available precipitation. 
        """
        self.t +=1

        a = self.a
        b = self.b
        c = self.c
        ymax = self.ymax
        wmax = self.wmax
        unit_area = self.unit_area
        crop_options = self.model.crop_options

        ### Yield calculation
        irr_depth = irr_depth.copy()[:,[0]]
        prec_aw_ = np.ones(irr_depth.shape)
        for ci, crop in enumerate(crop_options):
            prec_aw_[ci, :] = prec_aw[crop]

        w = irr_depth + prec_aw_
        w = w * i_crop
        w_ = w/wmax    #normalized applied water
        w_ = np.minimum(w_, 1)
        y_ = (a * w_**2 + b * w_ + c)   #normalized yield
        y_ = np.maximum(0, y_)
        y_ = y_ * i_crop
        
        self.update_crops(i_crop)   # update pre_i_crop
        
        y = y_ * ymax * unit_area * 1e-4      # 1e4 bu

        cm2m = 0.01
        v_c = irr_depth * unit_area * cm2m    # m-ha
        irr_vol = np.sum(v_c)                 # m-ha
        avg_y_y = np.sum(y_)
        avg_w = np.sum(w)

        # record
        self.y = y # 1e4 bu
        self.w = avg_w
        self.yield_rate_per_field = avg_y_y
        self.irr_vol_per_field = irr_vol     # m-ha

        return y, avg_y_y, irr_vol


class Field_1f1w_ci(mesa.Agent):
    """ Simulate a field agent in the model. """

    def __init__(self, unique_id, model, settings: dict, **kwargs):
        """Initialize a Field agent in the Mesa model."""
        # MESA required attributes => (unique_id, model)
        super().__init__(unique_id, model)
        self.agt_type = "Field"

        # Initialize attributes
        self.load_settings(settings)

        # Initialize and update crop
        crop_options = self.model.crop_options
        i_crop = np.zeros((self.n_c, 1))
        ini_crop = self.init["crop"]
        self.crop = ini_crop
        i_c = crop_options.index(ini_crop)
        i_crop[i_c, 0] = 1
        self.i_crop = i_crop
        self.update_crops(i_crop)

        # Initialize field type
        self.field_type = self.init["field_type"]

        # Initialize aph_yield_records & aph_yield_dict (in unit of 1e4 bu/field)
        if self.model.activate_ci:
            self.aph_yield_dict = self.init.get("aph_yield")
            self.aph_yield_records = {
                "irrigated": {
                    c: [v] * 5 for c, v in self.aph_yield_dict["irrigated"].items()
                },
                "rainfed": {
                    c: [v] * 5 for c, v in self.aph_yield_dict["rainfed"].items()
                },
            }
            # Note that the premium_dict_for_dm will be populated in the behavior
            # module before optimization.
            self.premium_dict_for_dm = {
                "irrigated": {c: None for c in crop_options},
                "rainfed": {c: None for c in crop_options},
            }
        else:
            self.aph_yield_dict = None
            self.aph_yield_records = None
            self.premium_dict_for_dm = None

        # Additional attributes from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Initialize other variables
        self.t = 0
        self.irr_vol = None
        self.yield_rate_per_field = None  # Averaged value across a field
        self.irr_vol_per_field = None  # Averaged value across a field

    def load_settings(self, settings: dict):
        """
        Load the field settings from a dictionary.

        Parameters
        ----------
        settings : dict
            A dictionary containing settings for the field, including field
            area, water yield curves for crops, and technological coefficients.
        """
        crop_options = self.model.crop_options
        self.field_area = settings["field_area"]
        self.water_yield_curves = settings["water_yield_curves"]
        self.prec_aw_id = settings["prec_aw_id"]
        self.init = settings["init"]

        self.county = settings.get("county")

        self.n_c = len(crop_options)

        crop_par = np.array([self.water_yield_curves[c] for c in crop_options])
        self.ymax = crop_par[:, 0].reshape((-1, 1))  # (n_c, 1)
        self.wmax = crop_par[:, 1].reshape((-1, 1))  # (n_c, 1)
        self.a = crop_par[:, 2].reshape((-1, 1))  # (n_c, 1)
        self.b = crop_par[:, 3].reshape((-1, 1))  # (n_c, 1)
        self.c = crop_par[:, 4].reshape((-1, 1))  # (n_c, 1)
        try:
            self.min_y_ratio = crop_par[:, 5].reshape((-1, 1))  # (n_c, 1)
        except:
            self.min_y_ratio = np.zeros((self.n_c, 1))

        self.unit_area = self.field_area

    def update_crops(self, i_crop):
        """
        Update the crop types for each area split based on the given indicator
        array. The dimension of the array should be (n_c, 1).

        Parameters
        ----------
        i_crop : 2d array
            Indicator array representing the chosen crops for the next year.
            The dimension of the array should be (n_c, 1).

        Returns
        -------
        None
        """
        crop_options = self.model.crop_options
        # Use argmax instead of ==1 to avoid float numerical issue
        self.pre_i_crop = self.i_crop
        # crops = [crop_options[np.argmax(i_crop[s, :, 0])] for s in range(n_s)]
        crop = crop_options[np.argmax(i_crop[:, 0])]
        self.crop = crop
        self.i_crop = i_crop

    def update_aph_yield(self, field_type, crop_yield):
        """Update the aph_yield_records & aph_yield_dict for crop insurance.

        Should be triggered after the yield calculation and premium
        calculation in the finance after everthing is done.

        Parameters
        ----------
        field_type : str
            The type of the field ("irrigated" or "rainfed").
        crop_yield : float
            The yield of the crop [1e4 bu].
        """
        crop = self.crop
        self.aph_yield_records[field_type][crop].append(crop_yield)
        self.aph_yield_dict[field_type][crop] = np.mean(
            self.aph_yield_records[field_type][crop][-5:]
        )

    def step(self, irr_depth, i_crop, prec_aw: dict) -> tuple:
        """
        Perform a single step of field operation, calculating yields and
        irrigation volumes.

        Parameters
        ----------
        irr_depth : 3d array
            The depth of irrigation applied [cm]. Dimensions: (n_s, n_c, 1)
        i_crop : 3d array
            Indicator array representing the chosen crops for each area split.
            Dimensions: (n_s, n_c, 1).
        prec_aw : dict
            A dictionary of available precipitation for each crop.
            {"corn": 27.02, "sorghum": 22.81}

        Returns
        -------
        tuple
            A tuple containing yield [1e4 bu], average yield rate [-], and
            total irrigation volume [m-ha].

        Notes
        -----
        This method calculates the yield based on the applied irrigation, chosen crops,
        install technology, and available precipitation.
        """
        self.t += 1

        a = self.a
        b = self.b
        c = self.c
        ymax = self.ymax
        wmax = self.wmax
        unit_area = self.unit_area
        crop_options = self.model.crop_options

        ### Yield calculation
        irr_depth = irr_depth.copy()[:, [0]]
        prec_aw_ = np.ones(irr_depth.shape)
        for ci, crop in enumerate(crop_options):
            prec_aw_[ci, :] = prec_aw[crop]

        w = irr_depth + prec_aw_
        w = w * i_crop
        w_ = w / wmax  # normalized applied water
        w_ = np.minimum(w_, 1)
        y_ = a * w_**2 + b * w_ + c  # normalized yield
        y_ = np.maximum(0, y_)
        y_ = y_ * i_crop

        self.update_crops(i_crop)  # update pre_i_crop

        y = y_ * ymax * unit_area * 1e-4  # 1e4 bu/field

        cm2m = 0.01
        v_c = irr_depth * unit_area * cm2m  # m-ha
        irr_vol = np.sum(v_c)  # m-ha
        avg_y_y = np.sum(y_)
        avg_w = np.sum(w)

        # record
        self.y = y
        self.w = avg_w
        self.yield_rate_per_field = avg_y_y
        self.irr_vol_per_field = irr_vol  # m-ha

        # All crop insurance related update should be done in the finance module.
        return y, avg_y_y, irr_vol

class Field_aquacrop(mesa.Agent):
    """ Simulate a field agent in the model, focusing on coupling with the Aquacrop model """
    def __init__(self, unique_id, model, settings: dict, **kwargs):
        """Initialize a Field agent.

        Parameters
        ----------
        unique_id : int
            A unique identifier for this agent.
        model
            The mesa model instance to which this agent belongs.
        settings : dict
            A dictionary containing initial settings for the field, which include field
            area, initial conditions, and crop options.
        """
        # MESA required attributes => (unique_id, model)
        super().__init__(unique_id, model)
        self.agt_type = "Field"

        # Initialize attributes
        self.load_settings(settings)

        # Initialize and update crop
        crop_options = self.model.crop_options
        ini_crop = self.init["crop"]
        self.crop = ini_crop
        self.i_crop = np.zeros((self.n_c, 1))
        i_c = crop_options.index(ini_crop)
        self.i_crop[i_c, 0] = 1
        self.update_crops(self.i_crop)
        
        # Initialize field type
        self.field_type = self.init["field_type"]

        # Additional attributes from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Initialize other variables
        self.t = 0
        self.irr_vol = None
        self.yield_rate_per_field = None    # Averaged value across a field 
        self.irr_vol_per_field = None       # Averaged value across a field 

    def load_settings(self, settings: dict):
        """
        Load the field settings from a dictionary.
    
        Parameters
        ----------
        settings : dict
            A dictionary containing settings for the field, including field area and initial conditions
        """
        self.field_area = settings["field_area"]
        self.init = settings["init"]
        self.n_c = len(self.model.crop_options)

        # Crop parameters are no longer needed for CSV generation
        self.unit_area = self.field_area

    def update_crops(self, i_crop):
        """
        Update the crop types based on the given indicator array.

        Parameters
        ----------
        i_crop : 2d array
            Indicator array representing the chosen crops for the next year. 
            The dimension of the array should be (n_c, 1).

        Returns
        -------
        None
        """
        crop_options = self.model.crop_options
        self.pre_i_crop = self.i_crop
        crop = crop_options[np.argmax(i_crop[:, 0])]
        self.crop = crop
        self.i_crop = i_crop

    def step(self, irr_depth, i_crop, prec_aw: dict) -> tuple:
        """
        Perform a single step of field operation, preparing data for coupling with Aquacrop
    
        Parameters
        ----------
        irr_depth : 3d array
            The depth of irrigation applied [cm]. Dimensions: (n_s, n_c, 1)
        i_crop : 3d array
            Indicator array representing the chosen crops for each area split.
            Dimensions: (n_s, n_c, 1).
        prec_aw : dict
            A dictionary of available precipitation for each crop. 
            {"corn": 27.02, "sorghum": 22.81}
    
        Returns
        -------
        tuple
            A tuple containing yield [1e4 bu], average yield rate [-], and 
            total irrigation volume [m-ha].
    
        Notes
        -----
        This method prepares data for the Aquacrop model by saving relevant information to a CSV file, including the maximum irrigation season, crop name, and irrigation method. 
        """
        self.t += 1

        # Calculate total irrigation volume
        cm2m = 0.01
        v_c = irr_depth * self.unit_area * cm2m  # m-ha
        irr_vol = np.sum(v_c)  # m-ha

        # Prepare data for CSV output
        max_irrseason = irr_depth.flatten().tolist()
        crop_name = [self.crop]  # single crop, no need for flatten
        irrig_method = [self.field_type]  # assuming this is for irrigation method

        # Define the path to the CSV file
        working_directory = "/path/to/working/directory"
        folder_name = "examples"
        file_name = "existing_data.csv"
        file_path = os.path.join(working_directory, folder_name, file_name)

        print(f"CSV file path: {file_path}")  # Debugging: Print file path

        # Check if the file exists
        if os.path.exists(file_path):
            # Load existing data
            df_existing = pd.read_csv(file_path)
            print(f"Existing data:\n{df_existing.head()}")  # Debugging: Print existing data

            # Create new columns with the data to append
            new_data = pd.DataFrame({
                'max_irrseason': [max_irrseason],
                'crop_name': crop_name,
                'irrig_method': irrig_method
            })
            
            # Append new data to the existing DataFrame
            df_updated = pd.concat([df_existing, new_data], ignore_index=True)
        else:
            # If file does not exist, create a new DataFrame
            df_updated = pd.DataFrame({
                'max_irrseason': [max_irrseason],
                'crop_name': crop_name,
                'irrig_method': irrig_method
            })

        # Save updated DataFrame back to the CSV file
        df_updated.to_csv(file_path, index=False)
        print(f"Data saved to CSV.")  # Debugging: Confirm data save

        return irr_vol, self.crop