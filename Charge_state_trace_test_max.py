# coding=utf-8
from pi3diamond import pi3d
import numpy as np
import os
import UserScripts.helpers.sequence_creation_helpers as sch; reload(sch)
import UserScripts.helpers.shared as shared; reload(shared)
import multi_channel_awg_seq as MCAS; reload(MCAS)
import UserScripts.helpers.snippets_awg as sna; reload(sna)
import UserScripts.helpers.shared as ush;reload(ush)
from qutip_enhanced import *
import AWG_M8190A_Elements as E


from collections import OrderedDict
# import necessary modules

seq_name = os.path.basename(__file__).split('.')[0]
nuclear = sch.create_nuclear(__file__)
with open(os.path.abspath(__file__).split('.')[0] + ".py", 'r') as f:
    meas_code = f.read()

__TAU_HALF__ = 2*192/12e3

ael = 1.0

def ret_ret_mcas(pdc):
    def ret_mcas(current_iterator_df):
        mcas = MCAS.MultiChSeq(seq_name=seq_name, ch_dict={'2g': [1, 2], '128m': [1, 2]})
        for idx, _I_ in current_iterator_df.iterrows():
            n = _I_['ssr_reps']

            if _I_['state_result'] == 'charge_state':
                #sna.ssr(mcas, nuc='charge_state', step_idx = 1)
                #sna.polarize_green(mcas, new_segment=True)
                freq = pi3d.tt.mfl({'14N': [+1]}, ms_trans=_I_['ms'])
                sna.ssr(mcas, frequencies=freq, wait_dur=0.0, robust=False, nuc='charge_state', mixer_deg=-90, step_idx=0, laser_dur=float(_I_['laser_dur_1']))
                pd128m2 = dict(sync_marker=True)
                mcas.asc(length_mus=0.175, pd128m2=pd128m2)
                mcas.dynamic_control = True

            pi3d.gated_counter.set_n_values(mcas)
        return mcas
    return ret_mcas


#def nuclear_rabi(mcas, new_segment=False, **kwargs):
#    type = 'robust' if 'wave_file' in kwargs else 'sine'
#    if new_segment:
#        mcas.start_new_segment(name='nuclear_rabi')
#    if 'pd128m' in kwargs:
#        raise Exception('Error!')
#    pd = {}
#    for awg_str, chl in mcas.ch_dict.items():
#        for ch in chl:
#            if 'pd' + awg_str + str(ch) in kwargs:
#                pd['pd' + awg_str + str(ch)] = kwargs.pop('pd' + awg_str + str(ch), None)
#    mcas.asc(pd128m1=dict(type=type, **kwargs), name=kwargs.get('name', 'rf'), **pd)


def settings(pdc={}):
    ana_seq=[
        #['init', '>', 8, 10, 5, 1],
        ['result', '>', 8, 5, 5, 1], # 3 is repetitions
        #['result', '<', 0, 0, 10, 3],
        #['result', '<', 0, 0, 10, 3]
        #['result', '<', 0, 0, 10, 2]
        #['result', '<', 'auto', 123123, 1, 1],
    ]
    sch.settings(
        nuclear=nuclear,
        ret_mcas=ret_ret_mcas(pdc),
        analyze_sequence=ana_seq,
        pdc=pdc,
        meas_code=meas_code
    )
    nuclear.x_axis_title = 'tau_half [mus]'
    nuclear.analyze_type = 'consecutive'
    #nuclear.analyze_type = 'standard'


    pi3d.gated_counter.trace.analyze_type = 'consecutive'
    pi3d.gated_counter.trace.consecutive_valid_result_numbers = [0]
    pi3d.gated_counter.trace.average_results = True

    nuclear.parameters = OrderedDict(
        (
            ('sweeps', range(2)),
            #('rabi_period', [0.1,0.2]),
            #('state_result', ['n+','+','nn+']),
            #('state_result', ['+']),
            ('state_result', ['charge_state']),
            ('laser_dur_1', [800]),  # you can set duration variable
            #('laser_dur_1', [1000]),  # you can set duration variable  E.round_length_mus_full_sample(np.logspace(0, 5, 20)
            #('laser_dur_2', [20000]),


            ('ms', [-1]),
            ('ddt', ['hahn']),# 'fid','hahn', 'xy4', 'xy16', 'kdd4', 'kdd16']),
            ('n_rep_dd', [1]),
            ('ssr_reps',[1500]),
            ('orange_laser_power', [0.1])  # uW
        )
    )
    nuclear.number_of_simultaneous_measurements = 1#len(nuclear.parameters['phase_pi2_2'])

def run_fun(abort, **kwargs):
    pi3d.readout_duration = 150e6
    nuclear.debug_mode = False
    nuclear.dynamic_charge_init = True
    settings()
    nuclear.run(abort)
