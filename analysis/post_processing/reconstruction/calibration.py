import numpy as np

from mlreco.utils.globals import TRACK_SHP
from mlreco.utils.calibration import CalibrationManager

from analysis.post_processing import PostProcessor


class CalibrationProcessor(PostProcessor):
    '''
    Apply calibrations to the reconstructed particles.
    '''
    name = 'apply_calibrations'
    data_cap_opt = ['run_info']
    result_cap = ['particles']

    def __init__(self,
                 run_mode='both',
                 dedx=2.2,
                 do_tracking=False,
                 **cfg):
        '''
        Initialize the calibration manager.

        Parameters
        ----------
        dedx : float, default 2.2
            Static value of dE/dx used to compute the recombination factor
        do_tracking : bool, default False
            Segment track to get a proper local dQ/dx estimate
        **cfg : dict
            Calibration manager configuration
        '''
        # Initialize the parent class
        super().__init__(run_mode)

        # Initialize the calibrator
        self.calibrator = CalibrationManager(cfg)
        self.dedx = dedx
        self.do_tracking = do_tracking

    def process(self, data_dict, result_dict):
        '''
        Apply calibrations to all particles in one entry.

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        '''
        # Fetch the run info
        run_id = None
        if 'run_info' in data_dict:
            run_info = data_dict['run_info'][0] # TODO: Why? Get rid of index
            run_id = run_info['run']

        # Loop over particle objects
        for k in self.part_keys:
            for p in result_dict[k]:
                # Make sure the particle coordinates are expressed in cm
                self.check_units(p)

                # Get point coordinates
                points = self.get_points(p)
                if not len(points):
                    continue

                # Apply calibration
                if not self.do_tracking or p.semantic_type != TRACK_SHP:
                    depositions = self.calibrator.process(
                            points, p.depositions, p.sources, run_id, self.dedx)
                else:
                    depositions = self.calibrator.process(
                            points, p.depositions, p.sources, run_id, track=True)

                p.depositions = depositions

        return {}, {}
