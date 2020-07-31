from __future__ import print_function, absolute_import, division
__metaclass__ = type

import sys
if sys.version_info.major == 2:
    from imp import reload
else:
    from importlib import reload

from pi3diamond import pi3d
import zipfile
import time
import misc; reload(misc)
import traceback
import datetime
import os
import numpy as np
import pandas as pd
import logging
import collections

from numbers import Number
from qutip_enhanced.data_generation import DataGeneration
from qutip_enhanced.util import ret_property_list_element
from qutip_enhanced import save_qutip_enhanced
import qutip_enhanced.data_handling as data_handling
from collections import OrderedDict

class NuclearOPs(DataGeneration):

    def __init__(self):
        super(NuclearOPs, self).__init__()
        self.odmr_pd = dict(
            n=0,
            freq=None,
            size={'left': '3', 'right': ''},
            repeat=False,
        )
        self.odmr_pd_refocus = dict(
            n=1,
            freq=None,
            size={'left': '3', 'right': ''},
            repeat=False,
        )


    state = ret_property_list_element('state', ['idle', 'run', 'sequence_testing', 'sequence_debug_interrupted', 'sequence_ok'])

    # Tracking stuff:
    refocus_interval = misc.ret_property_typecheck('refocus_interval', int)
    odmr_interval = misc.ret_property_typecheck('odmr_interval', Number)
    additional_recalibration_interval = misc.ret_property_typecheck('additional_recalibration_interval', int)

    __TITLE_DATE_FORMAT__ = '%Y%m%dh%Hm%Ms%S'

    @property
    def ana_trace(self):
        return pi3d.gated_counter.trace

    @property
    def analyze_type(self):
        try:
            return self.ana_trace.analyze_type
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @analyze_type.setter
    def analyze_type(self, val):
        self.ana_trace.analyze_type = val

    @property
    def number_of_simultaneous_measurements(self):
        return self.ana_trace.number_of_simultaneous_measurements

    @number_of_simultaneous_measurements.setter
    def number_of_simultaneous_measurements(self, val):
        self.ana_trace.number_of_simultaneous_measurements = val

    @property
    def observation_names(self):
        try:
            if hasattr(self, '_observation_names'):
                return self._observation_names
            else:
                return ['result_{}'.format(i) for i in range(self.number_of_results)] + ['trace', 'events', 'thresholds', 'start_time', 'end_time', 'local_oscillator_freq', 'confocal_x', 'confocal_y', 'confocal_z']
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def dtypes(self):
        if not hasattr(self, '_dtypes'):
            self._dtypes = dict(trace='object', events='int', start_time='datetime', end_time='datetime', local_oscillator_freq='float', thresholds='object', confocal_x='float', confocal_y='float', confocal_z='float')
        return self._dtypes

    @property
    def number_of_results(self):
        return self.ana_trace.number_of_results

    def run(self, *args, **kwargs):
        if getattr(self, 'debug_mode', False):
            self.run_debug_sequence(*args, **kwargs)
        else:
            self.run_measurement(*args, **kwargs)

    def run_measurement(self, abort, **kwargs):
        self.init_run(**kwargs)
        self.df_refocus_pos = pd.DataFrame(OrderedDict(confocal_x=[pi3d.confocal.x], confocal_y=[pi3d.confocal.y], confocal_z=[pi3d.confocal.z]))
        try:
            pi3d.microwave.On()
            self.do_refocusodmr(abort, check_odmr_frequency_drift_ok=False, initial_odmr=False)
            for idx, _ in enumerate(self.iterator()):
                if abort.is_set(): break
                while True:
                    if abort.is_set(): break
                    self.setup_rf(self.current_iterator_df)
                    if abort.is_set(): break
                    self.data.set_observations([OrderedDict(local_oscillator_freq=pi3d.tt.current_local_oscillator_freq)]*self.number_of_simultaneous_measurements)
                    self.data.set_observations(pd.concat([self.df_refocus_pos.iloc[-1:, :]]*self.number_of_simultaneous_measurements).reset_index(drop=True))
                    self.data.set_observations([OrderedDict(start_time=datetime.datetime.now())]*self.number_of_simultaneous_measurements)
                    # insert here the dynamic control
                    self.get_trace(abort)
                    if abort.is_set(): break
                    self.data.set_observations([OrderedDict(end_time=datetime.datetime.now())]*self.number_of_simultaneous_measurements)
                    self.data.set_observations([OrderedDict(trace=self.ana_trace.trace)]*self.number_of_simultaneous_measurements)
                    if abort.is_set(): break
                    repeat_measurement = self.analyze()
                    if abort.is_set(): break
                    odmr_frequency_drift_ok = self.do_refocusodmr(abort=abort)
                    if odmr_frequency_drift_ok and not repeat_measurement:
                        break
                if hasattr(self, '_pld'):
                    self.pld.new_data_arrived()
                if abort.is_set(): break
                self.save()
        except Exception:
            abort.set()
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)
        finally:
            self.data._df = data_handling.df_take_duplicate_rows(self.data.df, self.iterator_df_done) #drops unfinished measurements,
            self.pld.new_data_arrived()
            pi3d.multi_channel_awg_sequence.stop_awgs(pi3d.awgs)
            self.state = 'idle'
            self.update_current_str()
            if self.session_meas_count == 0:
                self.pld.gui.close_gui()
                if hasattr(self.data, 'init_from_file') and self.data.init_from_file is not None:
                    self.move_init_from_file_folder_back()
            if os.path.exists(self.save_dir) and not os.listdir(self.save_dir):
                os.rmdir(self.save_dir)


    @property
    def session_meas_count(self):
        if len(self.data.df) == 0 or len(self.iterator_df_done) == 0:
            return 0
        else:
            return len(self.iterator_df_done) - len(self.data.df[(self.data.df.start_time < self.start_time) & (self.data.df.start_time > datetime.datetime(1900, 1, 1))])


    def run_debug_sequence(self, abort, **kwargs):
        if any([key in kwargs for key in ['iff', 'init_from_file']]):
            raise Exception('Error: Data initialization from file (.hdf or .csv) not allwoed in sequence debug mode.')
        if len(self.parameters['sweeps']) != 1:
            print('Debug mode, number of sweeps set to one.')
            self.parameters['sweeps'] = [0]
        self.init_run(**kwargs)
        self.state = 'sequence_testing'
        try:
            pi3d.mcas_dict.debug_mode = True
            for idx, _ in enumerate(self.iterator()):
                if abort.is_set(): break
                self.data.set_observations([OrderedDict(start_time=datetime.datetime.now())] * self.number_of_simultaneous_measurements)
                self.setup_rf(self.current_iterator_df)
                self.data.set_observations([OrderedDict(end_time=datetime.datetime.now())] * self.number_of_simultaneous_measurements)
            if not abort.is_set():
                self.state = 'sequence_ok'
        except Exception:
            self.state = 'sequence_debug_interrupted'
            abort.set()
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)
        finally:
            pi3d.mcas_dict.debug_mode = False
            pi3d.multi_channel_awg_sequence.stop_awgs(pi3d.awgs)
            self.update_current_str()
            if os.path.exists(self.save_dir) and not os.listdir(self.save_dir):
                os.rmdir(self.save_dir)

    def confocal_pos_moving_average(self, n):
        return self.df_refocus_pos[['confocal_x', 'confocal_y', 'confocal_z']].rolling(n, win_type='boxcar', center=True).sum().dropna()/n

    @property
    def refocus_moving_average_num(self):
        return getattr(self, '_refocus_moving_average_num', 10)

    @refocus_moving_average_num.setter
    def refocus_moving_average_num(self, val):
        self._refocus_moving_average_num = val

    @property
    def sweeps(self):
        return self.parameters['sweeps']

    def run_refocus(self):
        pi3d.confocal.run_refocus()
        self.df_refocus_pos = self.df_refocus_pos.append(pd.DataFrame(OrderedDict(confocal_x=[pi3d.confocal.x], confocal_y=[pi3d.confocal.y], confocal_z=[pi3d.confocal.z]))).reset_index(drop=True)
        if self.refocus_moving_average_num > 1:
            ma = self.confocal_pos_moving_average(min(len(self.df_refocus_pos), self.refocus_moving_average_num))
            for axis in ['x', 'y', 'z']:
                setattr(pi3d.confocal, axis, getattr(ma, 'confocal_{}'.format(axis)).iloc[-1])
            logging.getLogger().info("Refocus ma_deviation [nm]: {}, {}, {}".format(*[(getattr(pi3d.confocal, axis) - self.df_refocus_pos.iloc[-1, :]['confocal_{}'.format(axis)])*1000 for axis in ['x', 'y', 'z']]))

    def add_odmr_script_to_queue(self, abort, pd):
        sys.modules[pi3d.init_task(name='refocus_confocal_odmr', folder='D:/Python/pi3diamond/UserScripts/')].run_fun(abort=abort, **pd)

    def do_refocusodmr(self, abort=None, check_odmr_frequency_drift_ok=True, initial_odmr=False):
        if abort.is_set():
            logging.getLogger().info('do_refocusodmr stopped here0')
        pi3d.odmr.file_name = self.file_name
        delta_t = time.time() - self.last_odmr
        if self.odmr_interval != 0 and (delta_t >= self.odmr_interval) or len(self.data.df) == 0 or initial_odmr:
            if check_odmr_frequency_drift_ok and hasattr(self, 'maximum_odmr_drift'):
                self.add_odmr_script_to_queue(abort, self.odmr_pd)
                current_drift = np.abs(pi3d.tt.current_local_oscillator_freq - self.data.df.iloc[-1, :].local_oscillator_freq)
                if current_drift > self.maximum_odmr_drift:
                    logging.getLogger().info("Too much drift ({} > {}), trying again!".format(current_drift, self.maximum_odmr_drift))
                    odmr_frequency_drift_ok = False
                else:
                    logging.getLogger().info("Drift is ok  ({} < {})".format(current_drift, self.maximum_odmr_drift))
                    odmr_frequency_drift_ok = True
                if self.refocus_interval != 0 and self.odmr_count % self.refocus_interval == 0:
                    self.add_odmr_script_to_queue(abort, self.odmr_pd_refocus)
            else:
                if self.refocus_interval != 0 and self.odmr_count % self.refocus_interval == 0:
                    self.add_odmr_script_to_queue(abort, self.odmr_pd_refocus)
                else:
                    self.add_odmr_script_to_queue(abort, self.odmr_pd)
                odmr_frequency_drift_ok = True
            self.odmr_count += 1
            self.last_odmr = time.time()
            if abort.is_set():
                logging.getLogger().info('do_refocusodmr stopped here1')
            return odmr_frequency_drift_ok
        elif check_odmr_frequency_drift_ok:
            if abort.is_set():
                logging.getLogger().info('do_refocusodmr stopped here2')
            return True

    def odmr_frequency_drift_ok(self):
        if not hasattr(self, 'maximum_odmr_drift'):
            return True
        if len(self.data.df) > 0:
            current_drift = np.abs(pi3d.tt.current_local_oscillator_freq - self.data.df.iloc[-1,:].local_oscillator_freq)
            if current_drift > self.maximum_odmr_drift:
                logging.getLogger().info("Too much drift ({} > {}), trying again!".format(current_drift, self.maximum_odmr_drift))
                return False
            else:
                logging.getLogger().info("Drift is ok  ({} > {})".format(current_drift, self.maximum_odmr_drift))
                return True

    def reinit(self):
        super(NuclearOPs, self).reinit()
        self.odmr_count = 0
        self.additional_recalibration_interval_count = 0
        self.last_odmr = time.time()
        self.last_rabi_refocus = time.time()


    def get_trace(self, abort):
        self.mcas.initialize()
        pi3d.gated_counter.count(abort, ch_dict=self.mcas.ch_dict)

    def setup_rf(self, current_iterator_df):
        self.mcas = self.ret_mcas(current_iterator_df)
        pi3d.mcas_dict[self.mcas.name] = self.mcas

    def analyze(self, data=None, ana_trace=None, start_idx=None):
        if ana_trace is None:
            ana_trace = self.ana_trace
            if self.analyze_type != ana_trace.analyze_type:
                raise Exception('This was supposed to be a sanity check. The programmer made shit.')
        data = self.data if data is None else data
        if ana_trace.analyze_type is not None:
            df = ana_trace.analyze().df
            if (df.events == 0).any() and not self.analyze_type == 'consecutive' and df.at[0, 'events'] != 0:
                return True
            if 'result_num' in df.columns: #if results are not averaged
                obs_r = df.pivot_table(values='result', columns='result_num', index='sm').rename(columns=collections.OrderedDict([(i, 'result_{}'.format(i)) for i in df.result_num.unique()]))
            else:
                obs_r = df.rename(columns={'result': 'result_0'}).drop(columns=['step', 'events', 'sm'])
            data.set_observations(obs_r, start_idx=start_idx)
            data.set_observations(df.groupby(['sm']).agg({'events': np.sum}), start_idx=start_idx)
            data.set_observations(df.groupby(['sm']).agg({'thresholds': lambda x: [i for i in x]}), start_idx=start_idx)
            logging.getLogger().info(df)
            logging.getLogger().info(ana_trace.analyze_type)
            return False

    def reanalyze(self, do_while_run=False, **kwargs):
        if self.state == 'run' and not do_while_run:
            print('Measurement is running.\nReanalyzation will write to data.df and may interfere with the running measurement doing the same.\nIf you want to reanalyze anyway, pass argument do_while_run=True')
            return
        import Analysis
        ana_trace = Analysis.Trace()
        for key in ['analyze_type', 'number_of_simultaneous_measurements', 'analyze_sequence', 'binning_factor', 'average_results', 'consecutive_valid_result_numbers']:
            setattr(ana_trace, key, kwargs.get(key, getattr(self.ana_trace, key)))
        for idx, _I_ in self.data.df.iterrows():
            if (idx-1)%ana_trace.number_of_simultaneous_measurements:
                continue ## What is it for? (seems that it doing nothings.
            if type(_I_['trace']) != np.ndarray:
                print('Interrupted reanalyzation at dataframe index {}, as trace is not a numpy array.\nMaybe, this is trace has just not been measured yet?\nTotal length of dataframe is {}'.format(idx, len(self.data.df)))
                break
            ana_trace.trace = _I_['trace']
            self.analyze(ana_trace=ana_trace, start_idx=idx)

    def save(self):
        if len(self.iterator_df_done) > 0 and not(hasattr(self, 'do_save') and not self.do_save):
            t0 = time.time()
            super(NuclearOPs, self).save(notify=True) #### IMPORTANT
            pi3d.save_pi3diamond(destination_dir=self.save_dir)
            save_qutip_enhanced(destination_dir=self.save_dir)
            logging.getLogger().info("saved nuclear to '{} ({:.3f})".format(self.save_dir, time.time() - t0))

    def reset_settings(self):
        """
        Here only settings are changed that are not automatically changed during run()
        :return:
        """
        self.additional_recalibration_interval = 0
        self.ret_mcas = None
        self.mcas = None
        self.refocus_interval = 2
        self.odmr_interval = 15
        self.file_notes = ''
        self.thread = None