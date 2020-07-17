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
            #n = _I_['ssr_reps']

            #sna.polarize(mcas, new_segment=True) # polarization of electron spin
            #sna.polarize_green(mcas, new_segment=True)  # polarization of electron spin

            mcas_charge_init = sna.get_dynamic_nuclear_spin_init()

            if 'ms-1' in _I_['transition_list']:
                sna.single_robust_electron_pi(mcas,
                                              nuc='all',
                                              transition={'-1': 'left', '+1':'right'}[_I_['ms']],
                                              frequencies=pi3d.tt.mfl({'14n': [0]}, ms_trans=_I_['ms']),
                                              new_segment=True,
                                              )

            sna.nuclear_rabi(mcas,
                             new_segment=True,
                             amplitudes=[_I_['amp']], ## amplitude of RF
                             name=_I_['transition_list'],
                             frequencies=[pi3d.tt.t(_I_['transition_list']).current_frequency],
                             length_mus=_I_['tau'])

            if '13c90' in _I_['transition_list']:
                state_result = 'nn+'
            elif '13c414' in _I_['transition_list']:
                state_result = 'n+'
            elif '14n' in _I_['transition_list']:
                state_result = '+'

            if state_result == 'n+':
                sna.ssr(mcas, frequencies=[pi3d.tt.mfl({'14n': [+1, 0, -1], '13c414': [+.5]}),
                                           pi3d.tt.mfl({'14n': [+1, 0, -1], '13c414': [-.5]})], nuc='13c414', robust=True, mixer_deg=-90, step_idx=0)
            elif state_result== 'nn+':
                sna.ssr(mcas, frequencies=[pi3d.tt.mfl({'14n': [+1, 0, -1], '13c414': [+.5, -.5], '13c90': [+.5]}),
                                           pi3d.tt.mfl({'14n': [+1, 0, -1], '13c414': [+.5, -.5], '13c90': [-.5]})],
                                            nuc='13c90', robust=True, mixer_deg=-90, step_idx=0)
            elif state_result == '+':
                freq1 = pi3d.tt.mfl({'14N': [+1]}, ms_trans=_I_['ms'])
                freq2 = pi3d.tt.mfl({'14N': [0]}, ms_trans=_I_['ms'])
                #freq3 = pi3d.tt.mfl({'14N': [-1]}, ms_trans=_I_['ms'])
                sna.ssr(mcas, frequencies=[freq1, freq2], nuc='14N+1', robust=True, mixer_deg=-90, step_idx=0)

            elif _I_['state_result'] in ["".join(i) for i in itertools.product(['+', '0', '-'], ['+', '-'], ['+', '-'])]:
                sna.ssr_single_state(mcas, state=_I_['state_result'], step_idx=0)

            mcas.asc(length_mus=0.175, pd128m2=dict(sync_marker=True))

            pi3d.gated_counter.set_n_values(mcas)
        return [mcas_charge_init, mcas]
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
        ['result', '<', 0, 0, 10, 2],
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


    pi3d.gated_counter.trace.analyze_type = 'consecutive'
    pi3d.gated_counter.trace.consecutive_valid_result_numbers = [0]
    pi3d.gated_counter.trace.average_results = True

    nuclear.parameters = OrderedDict(
        (
            ('sweeps', range(4)),
            ('rabi_period', [0.1]),
            #('state_result', ['n+','+','nn+']),
            #('transition_list',['13c414 ms0', '13c414 ms-1','13c90 ms-1']),
            ('transition_list', ['14n+1 ms0']),
            ('amp', [1]),
            #('tau',E.round_length_mus_full_sample(np.linspace(0.5,100,40))),
            #('state_result', ["".join(i) for i in itertools.product(['+','0','-'], ['+', '-'], ['+', '-'])]+['+','n+','nn+']),
            # ('state_result', ["".join(i) for i in itertools.product(['+', '0'], ['+', '-'], ['+','-'])]),
            # ('state_init', ["".join(i) for i in itertools.product(['+'], ['+','-'], ['+','-'])]),
            #('state_init', ['+++']),
            #('state_result', ['+++', '0++']),
            ('ms', ['-1']),
            ('ddt', ['hahn']),# 'fid','hahn', 'xy4', 'xy16', 'kdd4', 'kdd16']),
            ('n_rep_dd', [1]),
            ('tau', E.round_length_mus_full_sample(np.linspace(0.1, 70, 50))),  ### time scale: mus
            #('ssr_reps',range(1000,2000,400))
        )
    )
    nuclear.number_of_simultaneous_measurements = 1 #len(nuclear.parameters['phase_pi2_2'])

def run_fun(abort, **kwargs):
    pi3d.readout_duration = 150e6
    nuclear.debug_mode = False
    nuclear.dynamic_control = True
    settings()
    nuclear.run(abort)



####################################################################################################
# predefined transition lists
####################################################################################################
#transition_list = ['14n+1 ms0', '14n-1 ms0', '14n+1 ms-1', '14n-1 ms-1', '14n+1 ms+1', '14n-1 ms+1',
#                   '13c414 ms0', '13c414 ms-1', '13c414 ms+1',
#                   '13c90 ms-1', '13c90 ms+1']
