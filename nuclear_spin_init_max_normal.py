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

            sna.polarize_green(mcas, new_segment=True)

            if _I_['polarize']:
                sna.init_13c(mcas, s='90', new_segment=True, state=_I_['state_init'][2])
                sna.init_13c(mcas, s='414', new_segment=True, state=_I_['state_init'][1])
                sna.init_14n(mcas, new_segment=True, mn=_I_['state_init'][0])


            sna.ssr_single_state(mcas, state=_I_['state_init'],  step_idx=0)

            wave_file_kwargs = dict(filepath=sna.wfpd_all_but_standard[_I_['state_init']], rp=pi3d.tt.rp('e_rabi', mixer_deg=-90))

            #freq = pi3d.tt.mfl({'14n': [+1], '13c414': [+.5], '13c90': [+.5]})


            freq1 = pi3d.tt.mfl({'14N': [+1]}, ms_trans=_I_['ms'])
            freq2 = pi3d.tt.mfl({'14N': [0]}, ms_trans=_I_['ms'])
            freq3 = pi3d.tt.mfl({'14N': [-1]}, ms_trans=_I_['ms'])
            sna.ssr(mcas, frequencies=[freq1, freq2], nuc='14N+1', robust=True, mixer_deg=-90, step_idx=1)

            """
            sna.ssr(mcas,
                transition='left',
                robust=True,
                laser_dur=sna.__LASER_DUR_DICT__.get(_I_['state_init'], sna.__LASER_DUR_DICT__['single_state']),
                mixer_deg=-90,
                nuc=_I_['nuc'],
                frequencies=[pi3d.tt.mfl({'14n': [0]})],
                wave_file_kwargs=wave_file_kwargs,
                step_idx=1)
            """

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
        ['init', '>', 8, 10, 5, 1],
        ['result', '<', 0, 0, 10, 2], # 3 is repetitions
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
    nuclear.analyze_type = 'standard'
    #nuclear.analyze_type = 'standard'


    pi3d.gated_counter.trace.analyze_type = 'standard'
    pi3d.gated_counter.trace.consecutive_valid_result_numbers = [0]
    pi3d.gated_counter.trace.average_results = True

    nuclear.parameters = OrderedDict(
        (
            ('polarize', [True]),
            ('sweeps', range(3)),
            ('ssr_repetitions_init', np.arange(100, 1900, 100)),
            ('ssr_repetitions_readout', np.arange(100, 1900, 100)),
            #('rabi_period', [0.1,0.2]),
            #('state_result', ['n+','+','nn+']),
            #('state_result', ['+']),
            ('state_init', ['+++']),
            ('nuc', ['14n']),
            ('ms', [0]),
            ('n_rep_dd', [1]),
        )
    )
    nuclear.number_of_simultaneous_measurements = 1#len(nuclear.parameters['phase_pi2_2'])

def run_fun(abort, **kwargs):
    pi3d.readout_duration = 150e6
    nuclear.debug_mode = False
    settings()
    nuclear.run(abort)
