import multi_channel_awg_seq as MCAS; reload(MCAS)
from PyDAQmx import Task
import PyDAQmx
from pi3diamond import pi3d
import os
import sys


def polarize_green(mcas, new_segment=False, length_mus=0.2, laser_delay=0.1, **kwargs):
    if new_segment:
        mcas.start_new_segment(name='polarize')
    pd = {}
    for awg_str, chl in mcas.ch_dict.items():
        for ch in chl:
            if 'pd' + awg_str + str(ch) in kwargs:
                if not awg_str in pd:
                    pd[awg_str] = {}
                pd[awg_str][ch] = kwargs.pop('pd' + awg_str + str(ch))
    mcas.asc(length_mus=length_mus, name='polarize', green=True, **pd)
    mcas.asc(length_mus=laser_delay, name='laserdelay', **pd)
    mcas.asc(length_mus=1.0, name='wait_cts', **pd)


__TT_TRIGGER_LENGTH__ = 10*192/12e3

class DynamicControl: 

    def __init__(self, normal_dyn_mode=True):
        if normal_dyn_mode:
            self.mcas_charge_init = self.create_charge_state_init_sequence()
            self.mcas_algo = self.create_algorithm()
            self.prepare_mcas_dict_for_dynamic_control()
            self.prepare_awg_for_dynamic_control()
            # self.start_microcontroller()

    def prepare_mcas_dict_for_dynamic_control(self): 
        self.empty_mcas_dict()
        pi3d.md[self.mcas_charge_init.name] = self.mcas_charge_init
        pi3d.md[self.mcas_algo.name] = self.mcas_algo
    
    def prepare_awg_for_dynamic_control(self):
        self.set_awg_in_dynamic_mode('2g')
        self.set_awg_in_dynamic_mode('128m')
        self.initilize_first_sequences()
        self.set_awgs_in_run_mode()

    def start_microcontroller(self):
        # starts the microcontroller & the microcontroller runs charge_init & algo in dynamic mode
        # we start the microcontroller through a pulse on GPIOA PIN1 (A01)
        set_analog_output_voltage()
        reset_analog_output_voltage()
        
    def empty_mcas_dict(self): 
        for key in pi3d.md:
            pi3d.md.__delitem__(key)

        pi3d.md.connect_to_awgs(False)


    def create_charge_state_init_sequence(self, with_trigger=False):
        # this sequence should just be stored in '2g' 
        mcas = MCAS.MultiChSeq(name='charge_init', ch_dict={'2g': [1, 2], '128m': [1, 2]})
        mcas.dynamic_control = True
        polarize_green(mcas, new_segment=True)
        mcas.start_new_segment('orange', loop_count=1)
        pd128m2 = dict(smpl_marker=True)
        mcas.add_step_complete(name='orange', length_mus=800, pd128m2=pd128m2, orange=True)  # anpassen

        return mcas

    def create_algorithm(self):
        pd128m2 = dict(sync_marker=True)
        mcas = MCAS.MultiChSeq(name='second_sequence', ch_dict={'2g': [1, 2], '128m':[1,2]})
        mcas.dynamic_control = True
        mcas.start_new_segment('red', loop_count=20)
        mcas.add_step_complete(name='red', length_mus=1000, pd128m2=pd128m2, red=True)  # anpassen
        return mcas


    def set_awg_in_dynamic_mode(self, awg):   # only use awg"2g"(master awg)
        pi3d.awgs[awg].ch[1].dynamic_mode = 1
        pi3d.awgs[awg].ch[2].dynamic_mode = 1

    def initilize_first_sequences(self):
        pi3d.md[self.mcas_charge_init.name].initialize()

    def set_awgs_in_run_mode(self):
        pi3d.awgs['2g'].run = True
        pi3d.awgs['128m'].run = True

    def send_threshold_to_microcontroller(self, threshold):
        print("threshold")
    
def set_analog_output_voltage():
    voltage=3
    voltage_task=Task()
    voltage_task.CreateAOVoltageChan("/Dev1/ao3", "", -10.0, 10.0, PyDAQmx.DAQmx_Val_Volts, None)
    voltage_task.StartTask()
    voltage_task.WriteAnalogScalarF64(1, 10.0, voltage, None)
    voltage_task.StopTask()

def reset_analog_output_voltage():
    voltage=0
    voltage_task=Task()
    voltage_task.CreateAOVoltageChan("/Dev1/ao3", "", -10.0, 10.0, PyDAQmx.DAQmx_Val_Volts, None)
    voltage_task.StartTask()
    voltage_task.WriteAnalogScalarF64(1, 10.0, voltage, None)
    voltage_task.StopTask()



def run_fun(abort, **kwargs):
    DynamicControl()