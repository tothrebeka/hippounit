from __future__ import print_function
from __future__ import division
from future import standard_library
standard_library.install_aliases()
#from builtins import str
from builtins import range
from quantities.quantity import Quantity
from quantities import mV, nA
import sciunit
from sciunit import Test,Score
try:
    from sciunit import ObservationError
except:
    from sciunit.errors import ObservationError
import hippounit.capabilities as cap
from sciunit.utils import assert_dimensionless# Converters.
from sciunit.scores import BooleanScore,ZScore # Scores.

try:
    import numpy
except:
    print("NumPy not loaded.")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from neuron import h
import collections
import efel
import os
import multiprocessing
import multiprocessing.pool
import functools
import math
from scipy import stats

import json
from hippounit import plottools
import collections


try:
    import pickle as pickle
except:
    import pickle
import gzip

try:
    import copy_reg
except:
    import copyreg

from types import MethodType

from quantities import mV, nA, ms, V, s

from hippounit import scores

def _pickle_method(method):
    func_name = method.__func__.__name__
    obj = method.__self__
    cls = method.__self__.__class__
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


try:
    copy_reg.pickle(MethodType, _pickle_method, _unpickle_method)
except:
    copyreg.pickle(MethodType, _pickle_method, _unpickle_method)


class BackpropagatingAPTest_OLM(Test):
    """
    Tests the mode and efficacy of back-propagating action potentials on the apical trunk.

    Parameters
    ----------
    config : dict
        dictionary loaded from a JSON file, containing the parameters of the simulation
    observation : dict
        dictionary loaded from a JSON file, containing the experimental mean and std values for the features to be tested
    force_run : boolean
        If True and the pickle files containing the model's response to the simulation exists, the simulation won't be run again, traces are loaded from the pickle file
    force_run_FindCurrentStim : boolean
        If True and the pickle file containing the adjusted current stimulus parameter exists, the search wont run again, data are loaded from the pickle file
    base_directory : str
        Results will be saved here
    show_plot : boolean
        If False, plots are not displayed but still saved
    save_all : boolean
        If False, only the JSON files containing the absolute feature values, the feature error scores and the final scores, and a log file are saved, but the figures and pickle files are not.
    trunk_origin : list
        first element : name of the section from which the trunk originates, second element : position on section (E.g. ['soma[5]', 1]). If not set by the user, the end of the default soma section is used.
    """

    def __init__(self, config = {},
                observation = {"mean_AP1_amp_dendpersoma_at_57um" : None,
                 "std_AP1_amp_dendpersoma_at_57um" : None}, 
                
                name="Back-propagating action potential test" ,
                force_run=False,
                force_run_FindCurrentStim=False,
                base_directory= None,
                show_plot=True,
                save_all = True,
                trunk_origin = None):

        observation = self.format_data(observation)

        Test.__init__(self, observation, name)

        self.required_capabilities += (cap.ReceivesDoubleSquareCurrentHyperpol_ProvidesResponse_MultipleLocations,
                                        cap.ProvidesRecordingLocationsOnTrunk, cap.ReceivesDoubleSquareCurrentHyperpol_ProvidesResponse, 
                 cap.ReceivesVoltageWaveform_ProvidesResponse_MultipleLocations)

        self.force_run = force_run
        self.force_run_FindCurrentStim = force_run_FindCurrentStim

        self.show_plot = show_plot
        self.save_all = save_all

        self.base_directory = base_directory
        self.path_temp_data = None #added later, because model name is needed
        self.path_figs = None
        self.path_results = None
        self.trunk_origin = trunk_origin

        self.logFile = None
        self.test_log_filename = 'test_log.txt'

        self.npool = multiprocessing.cpu_count() - 1

        self.config = config

        description = "Tests the mode and efficacy of back-propagating action potentials on the apical trunk."

    score_type = scores.ZScore_backpropagatingAP_OLM

    def format_data(self, observation):
        for key1, val1 in observation.items():
            for key, val in list(observation[key1].items()):
                try:
                    assert type(observation[key1][key]) is Quantity
                except Exception as e:
                    quantity_parts = val.split(" ")
                    number = float(quantity_parts[0])
                    units = " ".join(quantity_parts[1:])
                    observation[key1][key] = Quantity(number, units)
        return observation

    def spikecount(self, delay, duration, soma_trace):

        trace = {}
        traces=[]
        trace['T'] = soma_trace[0]
        trace['V'] = soma_trace[1]
        trace['stim_start'] = [delay]
        trace['stim_end'] = [delay + duration]
        traces.append(trace)

        traces_results = efel.getFeatureValues(traces, ['Spikecount_stimint'])

        spikecount = traces_results[0]['Spikecount_stimint']

        return spikecount

    def binsearch(self, model, stim_range, delay, dur, section_stim, loc_stim, section_rec, loc_rec):
        c_minmax = stim_range
        c_step_start = 0.01
        c_step_stop= 0.002

        found = False
        spikecounts = []
        amplitudes = []

        while c_step_start >= c_step_stop and not found:

            c_stim = numpy.arange(c_minmax[0], c_minmax[1], c_step_start)

            first = 0
            last = numpy.size(c_stim, axis=0)-1

            while first <= last and not found:

                midpoint = (first + last)//2
                amplitude = c_stim[midpoint]

                result=[]

                pool = multiprocessing.Pool(1, maxtasksperchild = 1)    # I use multiprocessing to keep every NEURON related task in independent processes

                traces= pool.apply(self.run_cclamp_on_soma, args = (model, amplitude, delay, dur, section_stim, loc_stim, section_rec, loc_rec))
                pool.terminate()
                pool.join()
                del pool

                spikecount = self.spikecount(delay, dur, traces)

                amplitudes.append(amplitude)
                spikecounts.append(spikecount)

                #if spikecount >= 10 and spikecount <=20:
                if spikecount == 15:
                    found = True
                else:
                    #if spikecount > 20:
                    if spikecount > 15:
                        last = midpoint-1
                    #elif spikecount < 10:
                    elif spikecount < 15:
                        first = midpoint+1
            c_step_start=c_step_start/2

        if not found:
            amp_index = min((p for p in range(len(spikecounts)) if spikecounts[p] != 0), key=lambda i: abs(spikecounts[i]-15.0)) # we choose the one that is nearest to 15, but not 0
            # print list(p for p in range(len(spikecounts)) if spikecounts[p] != 0) # this gives the indices where spikecount is not 0, then i takes up these values
            #print amp_index
            amplitude = amplitudes[amp_index]
            spikecount = spikecounts[amp_index]


        binsearch_result=[found, amplitude, spikecount]
        #print binsearch_result

        return binsearch_result

    def run_cclamp_on_soma(self, model, amp, hyperpol_amp, delay, dur, section_stim, loc_stim, section_rec, loc_rec):

        if self.base_directory:
            self.path_temp_data = self.base_directory + 'temp_data/' + 'backpropagating_AP/' + model.name + '/'
        else:
            self.path_temp_data = model.base_directory + 'temp_data/' + 'backpropagating_AP/'


        try:
            if not os.path.exists(self.path_temp_data) and self.save_all:
                os.makedirs(self.path_temp_data)
        except OSError as e:
            if e.errno != 17:
                raise
            pass

        file_name = self.path_temp_data + 'soma_traces' + str(amp) + '_nA.p'


        if self.force_run_FindCurrentStim or (os.path.isfile(file_name) is False):
            t, v = model.get_vm_hyperpol(amp, hyperpol_amp, delay, dur, section_stim, loc_stim, section_rec, loc_rec)
            if self.save_all:
                pickle.dump([t, v], gzip.GzipFile(file_name, "wb"))


        else:
            [t, v] = pickle.load(gzip.GzipFile(file_name, "rb"))

        return [t, v]

    def find_current_amp(self, model, delay, dur, section_stim, loc_stim, section_rec, loc_rec):

        print('Finding appropriate current step amplitude...')

        amps = numpy.arange(0.0, 1.1, 0.1)
        #amps= [0.0, 0.2, 0.8]
        #amps=[0.0,0.3,0.8]
        #amps=[0.0,0.2, 0.9]

        pool = multiprocessing.Pool(self.npool, maxtasksperchild=1)


        run_cclamp_on_soma_ = functools.partial(self.run_cclamp_on_soma, model, delay=delay, dur=dur, section_stim=section_stim, loc_stim=loc_stim, section_rec=section_rec, loc_rec=loc_rec)
        traces = pool.map(run_cclamp_on_soma_, amps, chunksize=1)

        pool.terminate()
        pool.join()
        del pool

        spikecounts = []
        _spikecounts = []
        amplitudes = []
        amplitude = None
        spikecount = None

        message_to_logFile = '' # as it can not be open here because then  multiprocessing won't work under python3

        for i in range(len(traces)):
            spikecounts.append(self.spikecount(delay, dur, traces[i]))

        if max(spikecounts) < 10:

            message_to_logFile += 'The model fired at ' + str(max(spikecounts)[0]) + ' Hz to ' + str(amps[-1]) + ' nA current step, and did not reach 10 Hz firing rate as supposed (according to Bianchi et al 2012 Fig. 1 B eg.)\n'
            message_to_logFile += "---------------------------------------------------------------------------------------------------\n"

            print('The model fired at ' + str(max(spikecounts)[0]) + ' Hz to ' + str(amps[-1]) + ' nA current step, and did not reach 10 Hz firing rate as supposed (according to Bianchi et al 2012 Fig. 1 B eg.)')
            amplitude = None

        else:
            for i in range(len(spikecounts)):

                if i != len(spikecounts)-1 and i!= 0:
                    if spikecounts[i] >= 10 and spikecounts[i] <= 20 and (spikecounts[i-1] <= spikecounts[i] and spikecounts[i+1] >= spikecounts[i]):
                        amplitudes.append(amps[i])
                        _spikecounts.append(spikecounts[i])
                    elif spikecounts[i] < 10 and spikecounts[i+1] > 20 and (spikecounts[i-1] <= spikecounts[i] and spikecounts[i+1] >= spikecounts[i]):
                        binsearch_result = self.binsearch(model, [amps[i], amps[i+1]], delay, dur, section_stim, loc_stim, section_rec, loc_rec)
                        amplitude = binsearch_result[1]
                        spikecount = binsearch_result[2]
                elif i==0:  #spikecount[i-1]  is the last element here
                    if spikecounts[i] >= 10 and spikecounts[i] <= 20 and spikecounts[i+1] >= spikecounts[i]:
                        amplitudes.append(amps[i])
                        _spikecounts.append(spikecounts[i])
                    elif spikecounts[i] < 10 and spikecounts[i+1] > 20 and spikecounts[i+1] >= spikecounts[i]:
                        binsearch_result = self.binsearch(model, [amps[i], amps[i+1]], delay, dur, section_stim, loc_stim, section_rec, loc_rec)
                        amplitude = binsearch_result[1]
                        spikecount = binsearch_result[2]
                elif i == len(spikecounts)-1: # there is no spikecounts[i+1] in this case
                    if spikecounts[i] >= 10 and spikecounts[i] <= 20 and spikecounts[i-1] <= spikecounts[i]:
                        amplitudes.append(amps[i])
                        _spikecounts.append(spikecounts[i])
        if len(amplitudes) > 1:
            amp_index = min(list(range(len(_spikecounts))), key=lambda i: abs(_spikecounts[i]-15.0)) # we choose the one that is nearest to 15
            amplitude = amplitudes[amp_index]
            spikecount = _spikecounts[amp_index]

        elif len(amplitudes) == 1:
            amplitude = amplitudes[0]
            spikecount = _spikecounts[0]
        # if len(amplitudes) remained 0, binsearch found an amplitude

        #print amplitude, spikecount

        if spikecount < 10 or spikecount > 20:

            message_to_logFile += 'WARNING: No current amplitude value has been found to which the model\'s firing frequency is between 10 and 20 Hz. The simulation is done using the current amplitude value to which the models fires at a frequency nearest to 15 Hz, but not 0 Hz. Current step amplitude: ' + str(amplitude) + 'nA, frequency: ' + str(spikecount) + 'Hz\n'
            message_to_logFile += "---------------------------------------------------------------------------------------------------\n"

            print('WARNING: No current amplitude value has been found to which the model\'s firing frequency is between 10 and 20 Hz. The simulation is done using the current amplitude value to which the models fires at a frequency nearest to 15 Hz, but not 0 Hz. Current step amplitude: ' + str(amplitude) + 'nA, frequency: ' + str(spikecount) + 'Hz\n')


        if self.base_directory:
            self.path_figs = self.base_directory + 'figs/' + 'backpropagating_AP/' + model.name + '/'
        else:
            self.path_figs = model.base_directory + 'figs/' + 'backpropagating_AP/'


        try:
            if not os.path.exists(self.path_figs) and self.save_all:
                os.makedirs(self.path_figs)
        except OSError as e:
            if e.errno != 17:
                raise
            pass


        plt.figure()
        plt.plot(amps, spikecounts, 'o')
        if amplitude is not None and spikecount is not None:
            plt.plot(amplitude, spikecount, 'o', label = "Chosen current amplitude")
        plt.ylabel('Spikecount')
        plt.xlabel('current amplitude (nA)')
        lgd=plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'Spikecounts_bAP' + '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')

        return amplitude, message_to_logFile

    def find_rheobase(self, model, hyperpol_amp, delay, dur, section_stim, loc_stim, section_rec, loc_rec):#to find the rheobase current 

        print('Finding rheobase current...')

        upper_bound = 1.0
        lower_bound = 0.0

        amps = numpy.arange(lower_bound, upper_bound, 0.1)

        message_to_logFile = '' # as it can not be open here because then  multiprocessing won't work under python3
        
        for amp in amps:
            # trace = run_cclamp_on_soma(self, model, amp = amp, delay=delay, dur=dur, section_stim=section_stim, loc_stim=loc_stim, section_rec=section_rec, loc_rec=loc_rec)
            pool = multiprocessing.Pool(1, maxtasksperchild = 1)    # I use multiprocessing to keep every NEURON related task in independent processes

            trace= pool.apply(self.run_cclamp_on_soma, args = (model, amp, hyperpol_amp, delay, dur, section_stim, loc_stim, section_rec, loc_rec))#itt kéne run_cclamp_on_soma_hyperpol 
            pool.terminate()
            pool.join()
            del pool

            spikecount = self.spikecount(delay, dur, trace)
            
            if amp == upper_bound and spikecount == 0:

                message_to_logFile += 'The model didn\'t fire even at ' + str(upper_bound) + ' nA current step\n'
                message_to_logFile += "---------------------------------------------------------------------------------------------------\n"

                print('The model didn\'t fire even at ' + str(upper_bound) + ' nA current step')
                amplitude = None

            if spikecount > 0:
                upper_bound = amp
                break
        
        
        # binary search
        
        depth = 1
        max_depth = 5
        precision = 0.005
        while depth < max_depth or abs(upper_bound - lower_bound) > precision:
            print('ciklus')
            middle_bound = upper_bound - abs(upper_bound - lower_bound) / 2.0

            pool = multiprocessing.Pool(1, maxtasksperchild = 1)    # I use multiprocessing to keep every NEURON related task in independent processes

            trace= pool.apply(self.run_cclamp_on_soma, args = (model, middle_bound, hyperpol_amp, delay, dur, section_stim, loc_stim, section_rec, loc_rec))
            pool.terminate()
            pool.join()
            del pool

            spikecount = self.spikecount(delay, dur, trace)
            if spikecount == 0:
                lower_bound = middle_bound
                upper_bound = upper_bound
                depth = depth + 1
            else:
                lower_bound = lower_bound
                upper_bound = middle_bound
                depth = depth + 1

        message_to_logFile += 'Rheobase current: ' + str(upper_bound) + ' nA\n'
        message_to_logFile += "---------------------------------------------------------------------------------------------------\n"

        return upper_bound, message_to_logFile


    def hyperpol_current(self, model, delay, dur, section_stim, loc_stim, section_rec, loc_rec):# find a hyperpolarizing current to stop spontaneous firing 
        
        upper_bound = -0.02
        lower_bound = -0.01

        h_amps = numpy.arange(lower_bound, upper_bound, -0.002)
        
        for h_amp in h_amps:
            pool = multiprocessing.Pool(1, maxtasksperchild = 1)    # I use multiprocessing to keep every NEURON related task in independent processes
            
            
            trace= pool.apply(self.run_cclamp_on_soma, args = (model, 0, h_amp, delay, dur, section_stim, loc_stim, section_rec, loc_rec))
            pool.terminate()
            pool.join()
            del pool
            print('h_amp: ' + str(h_amp))
            spikecount = self.spikecount(delay, dur, trace)
            if spikecount == 0:
                hyperpol_amp = h_amp
                break

        return hyperpol_amp

    def cclamp(self, model, amp, hyperpol_amp, delay, dur, section_stim, loc_stim, dend_locations):

        if self.base_directory:
            self.path_temp_data = self.base_directory + 'temp_data/' + 'backpropagating_AP/' + model.name + '/'
        else:
            self.path_temp_data = model.base_directory + 'temp_data/' + 'backpropagating_AP/'


        try:
            if not os.path.exists(self.path_temp_data) and self.save_all:
                os.makedirs(self.path_temp_data)
        except OSError as e:
            if e.errno != 17:
                raise
            pass

        file_name = self.path_temp_data + 'cclamp_' + str(amp) + '.p'

        traces = {}

        if self.force_run or (os.path.isfile(file_name) is False):
            t, v_stim, v = model.get_multiple_vm_hyperpol(amp, hyperpol_amp, delay, dur, section_stim, loc_stim, dend_locations)

            traces['T'] = t
            traces['v_stim'] = v_stim
            traces['v_rec'] = v #dictionary key: dendritic location, value : corresponding V trace of each recording locations
            if self.save_all:
                pickle.dump(traces, gzip.GzipFile(file_name, "wb"))

        else:
            traces = pickle.load(gzip.GzipFile(file_name, "rb"))

        return traces

    def waveform_vclamp(self, model, soma_AP1_waveform, delay, duration_waveform, section_stim, loc_stim, dend_locations):# sets the voltage clamp to the elements of a vector (waveform, not const)

        print('a')
        if self.base_directory:
            self.path_temp_data = self.base_directory + 'temp_data/' + 'backpropagating_AP/' + model.name + '/'
        else:
            self.path_temp_data = model.base_directory + 'temp_data/' + 'backpropagating_AP/'


        try:
            if not os.path.exists(self.path_temp_data) and self.save_all:
                os.makedirs(self.path_temp_data)
        except OSError as e:
            if e.errno != 17:
                raise
            pass

        print('b')
        file_name = self.path_temp_data + 'waveform_vclamp' + '.p'

        traces = {}
        
        if self.force_run or (os.path.isfile(file_name) is False):
            
            t, v_stim, v = model.get_voltage_waveform_multiple_vm(soma_AP1_waveform, delay, duration_waveform, section_stim, loc_stim, dend_locations)

            traces['T'] = t
            traces['v_stim'] = v_stim
            traces['v_rec'] = v #dictionary key: dendritic location, value : corresponding V trace of each recording locations
            if self.save_all:
                pickle.dump(traces, gzip.GzipFile(file_name, "wb"))

        else:
            traces = pickle.load(gzip.GzipFile(file_name, "rb"))

        return traces

    def extract_somatic_spiking_features(self, traces, delay, duration):

        # soma
        trace = {}
        traces_for_efel=[]
        trace['T'] = traces['T']
        trace['V'] = traces['v_stim']
        trace['stim_start'] = [delay]
        trace['stim_end'] = [delay + duration]
        traces_for_efel.append(trace)


        efel.setDoubleSetting('interp_step', 0.025)
        efel.setDoubleSetting('DerivativeThreshold', 40.0)

        traces_results = efel.getFeatureValues(traces_for_efel, ['inv_first_ISI','AP_begin_time', 'doublet_ISI'])

        return traces_results

    def extract_amplitudes(self, traces, delay, traces_results, actual_distances):

        #soma_AP_begin_indices = traces_results[0]['AP_begin_indices']
        soma_AP_begin_time = traces_results[0]['AP_begin_time']
        soma_first_ISI = traces_results[0]['doublet_ISI'][0]

        s_indices_AP1 = numpy.where(traces['T'] >= (soma_AP_begin_time[0]-1.0))
        stimstart_ind = numpy.where(traces['T'] >= 0)[0][0]    
        if 10 < soma_first_ISI:
            plus = 10
        else:
            plus = soma_first_ISI-3
        e_indices_AP1 = numpy.where(traces['T'] >= (soma_AP_begin_time[0]+plus))
        start_index_AP1 = s_indices_AP1[0][0]
        end_index_AP1 = e_indices_AP1[0][0]
        soma_ap1 = traces['v_stim'][stimstart_ind:end_index_AP1]

        plt.figure()
        plt.plot(traces['T'][stimstart_ind:end_index_AP1] , soma_ap1)

        duration = traces['T'][end_index_AP1] - traces['T'][stimstart_ind]
        print('stim_start...', traces['T'][stimstart_ind])
       
        features = collections.OrderedDict()

        for key, value in traces['v_rec'].items():
            features[key] = collections.OrderedDict()
            for k, v in traces['v_rec'][key].items():
                features[key][k] = collections.OrderedDict()

                features[key][k]['AP1_amp']= float(numpy.amax(traces['v_rec'][key][k][start_index_AP1:end_index_AP1]) - traces['v_rec'][key][k][start_index_AP1])*mV
                features[key][k]['actual_distance'] = actual_distances[k]
        
        AP_amp_soma = float(numpy.amax(traces['v_stim'][start_index_AP1:end_index_AP1]) - traces['v_stim'][start_index_AP1])*mV 
        print('Printing AP_amp_soma...', AP_amp_soma)            

        # zoom to fist AP
        plt.figure()
        plt.plot(traces['T'],traces['v_stim'], 'r', label = 'soma')
        for key, value in traces['v_rec'].items():
            for k, v in traces['v_rec'][key].items():
                #plt.plot(traces['T'],traces['v_rec'][i], label = dend_locations[i][0]+'('+str(dend_locations[i][1])+') at '+str(self.config['recording']['distances'][i])+' um')
                #plt.plot(traces['T'],traces['v_rec'][key], label = dend_locations[key][0]+'('+str(dend_locations[key][1])+') at '+str(key)+' um')
                plt.plot(traces['T'],traces['v_rec'][key][k], label = k[0]+'('+str(k[1])+') at '+str(actual_distances[k])+' um')

        plt.xlabel('time (ms)')
        plt.ylabel('membrane potential (mV)')
        plt.title('First AP')
        plt.xlim(traces['T'][start_index_AP1], traces['T'][end_index_AP1])
        lgd=plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'AP1_traces'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')
       
        return features, AP_amp_soma, soma_ap1, duration


    def plot_traces(self, model, traces, dend_locations, actual_distances):

        if self.base_directory:
            self.path_figs = self.base_directory + 'figs/' + 'backpropagating_AP/' + model.name + '/'
        else:
            self.path_figs = model.base_directory + 'figs/' + 'backpropagating_AP/'


        try:
            if not os.path.exists(self.path_figs) and self.save_all:
                os.makedirs(self.path_figs)
        except OSError as e:
            if e.errno != 17:
                raise
            pass

        print("The figures are saved in the directory: ", self.path_figs)

        plt.figure()
        plt.plot(traces['T'],traces['v_stim'], 'r', label = 'soma')
        for key, value in traces['v_rec'].items():
            for k, v in traces['v_rec'][key].items():
                #plt.plot(traces['T'],traces['v_rec'][i], label = dend_locations[i][0]+'('+str(dend_locations[i][1])+') at '+str(self.config['recording']['distances'][i])+' um')
                #plt.plot(traces['T'],traces['v_rec'][key], label = dend_locations[key][0]+'('+str(dend_locations[key][1])+') at '+str(key)+' um')
                plt.plot(traces['T'],traces['v_rec'][key][k], label = k[0]+'('+str(k[1])+') at '+str(actual_distances[k])+' um')

        plt.xlabel('time (ms)')
        plt.ylabel('membrane potential (mV)')
        plt.title('Traces')
        lgd=plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')


        if self.save_all:
            plt.savefig(self.path_figs + 'traces'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')


    def plot_features(self, model, features, AP_amp_soma, actual_distances,Na_block):

        observation = self.observation

        model_AP1_amps = numpy.array([])
        exp_mean_AP1_amps = numpy.array([])
        exp_std_AP1_amps = numpy.array([])

        distances = []
        dists = numpy.array(self.config['recording']['distances'])
        location_labels = []

        for key, value in features.items():

            if Na_block:
                exp_mean_AP1_amps = numpy.append(exp_mean_AP1_amps, observation['blocked_Na']['mean_AP1_amp_dendpersoma_at_'+str(key)+'um'])
                exp_std_AP1_amps = numpy.append(exp_std_AP1_amps, observation['blocked_Na']['std_AP1_amp_dendpersoma_at_'+str(key)+'um'])
            else:
                exp_mean_AP1_amps = numpy.append(exp_mean_AP1_amps, observation['active_Na']['mean_AP1_amp_dendpersoma_at_'+str(key)+'um'])
                exp_std_AP1_amps = numpy.append(exp_std_AP1_amps, observation['active_Na']['std_AP1_amp_dendpersoma_at_'+str(key)+'um'])


            for k, v in features[key].items() :
                distances.append(actual_distances[k])
                model_AP1_amps = numpy.append(model_AP1_amps, features[key][k]['AP1_amp'])
               # model_APlast_amps = numpy.append(model_APlast_amps, features[key][k]['APlast_amp'])
                location_labels.append(k[0]+'('+str(k[1])+')')

        plt.figure()
        for i in range(len(distances)):
            plt.plot(distances[i], model_AP1_amps[i], marker ='o', linestyle='none', label = location_labels[i])
        plt.plot(0,AP_amp_soma, marker ='o', linestyle='none', label = 'soma')

        plt.xlabel('Distance from soma (um)')
        plt.ylabel('AP1_amp (mV)')
        plt.title('AP1 amps prediction')
        lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')

        plt.figure()
        for i in range(len(distances)):
            plt.plot(distances[i], model_AP1_amps[i]/AP_amp_soma, marker ='o', linestyle='none', label = location_labels[i])
        plt.plot(0,AP_amp_soma/AP_amp_soma, marker ='o', linestyle='none', label = 'soma')
        plt.errorbar(dists, exp_mean_AP1_amps, yerr = exp_std_AP1_amps, marker='o', linestyle='none', label = 'experiment')

        plt.xlabel('Distance from soma (um)')
        plt.ylabel('AP1_amp_dend/AP_amp_soma')
        plt.title('dend/soma prediction')
        lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')

        if self.save_all:
            plt.savefig(self.path_figs + 'AP1_amps'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')

    def plot_results(self, observation, prediction, errors, model_name_bAP):

        # Mean absolute feature values plot
        distances = numpy.array(self.config['recording']['distances'])

        model_mean = numpy.array([])
        model_std = numpy.array([])
        exp_mean = numpy.array([])
        exp_std = numpy.array([])

        
        model_mean = numpy.array([prediction['active_Na']['model_AP1_amp_dendpersoma_at_57um']['mean'], prediction['blocked_Na']['model_AP1_amp_dendpersoma_at_57um']['mean']])
        model_std = numpy.array([prediction['active_Na']['model_AP1_amp_dendpersoma_at_57um']['std'], prediction['blocked_Na']['model_AP1_amp_dendpersoma_at_57um']['std']])
        exp_mean = numpy.array([observation['active_Na']['mean_AP1_amp_dendpersoma_at_57um'], observation['blocked_Na']['mean_AP1_amp_dendpersoma_at_57um']])
        exp_std = numpy.array([observation['active_Na']['std_AP1_amp_dendpersoma_at_57um'], observation['blocked_Na']['std_AP1_amp_dendpersoma_at_57um']])
        
        plt.figure()
        plt.errorbar([1,2] , model_mean, yerr = model_std, marker ='o', linestyle='none', label = model_name_bAP)
        plt.errorbar([1,2] , exp_mean, yerr = exp_std, marker='o', linestyle='none', label = 'experiment')
        
        plt.ylabel('AP_amp (mV)')
        plt.title('Exp and model mean')
        plt.xticks([1,2],['active_Na','blocked_Na'])
        lgd=plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'AP1_amp_means'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')
        


        # Plot of errors

        keys = []
        values = []

        #fig, ax = plt.subplots()
        plt.figure()
        for key1,value1 in errors.items():
            for key, value in errors[key1].items():
                keys.append(key1+'_'+key)
                values.append(value)
        y=list(range(len(keys)))
        y.reverse()
        #ax.set_yticks(y)
        #print keys
        #print values
        plt.plot(values, y, 'o')
        plt.yticks(y, keys)
        if self.save_all:
            plt.savefig(self.path_figs + 'bAP_errors'+ '.pdf', bbox_inches='tight')

    def validate_observation(self, observation):
        '''
        for key, value in observation.items():
            try:
                assert type(observation[key]) is Quantity
            except Exception as e:
                raise ObservationError(("Observation must be of the form "
                                        "{'mean':float*mV,'std':float*mV}"))
        '''
        pass

    def generate_prediction(self, model, verbose=False):
        """Implementation of sciunit.Test.generate_prediction."""

        efel.reset()

        if self.base_directory:
            self.path_results = self.base_directory + 'results/' + 'backpropagating_AP/' + model.name + '/'
        else:
            self.path_results = model.base_directory + 'results/' + 'backpropagating_AP/'

        try:
            if not os.path.exists(self.path_results):
                os.makedirs(self.path_results)
        except OSError as e:
            if e.errno != 17:
                raise
            pass


        global model_name_bAP
        model_name_bAP = model.name

        distances = self.config['recording']['distances']
        tolerance = self.config['recording']['tolerance']

        dend_locations, actual_distances = model.find_trunk_locations_multiproc(distances, tolerance, self.trunk_origin)
        #print dend_locations, actual_distances

        print('Dendritic locations to be tested (with their actual distances):', actual_distances)

        traces={}
        delay = self.config['stimulus']['delay']
        duration = self.config['stimulus']['duration']
        #amplitude = self.config['stimulus']['amplitude']

        prediction = collections.OrderedDict()

        plt.close('all') #needed to avoid overlapping of saved images when the test is run on multiple models

        hyperpol_amp = self.hyperpol_current(model, 0, duration, "soma", 0.5, "soma", 0.5)
        print('Hyperpol amplitude: '+ str(hyperpol_amp) )

        amplitude, message_to_logFile = self.find_rheobase(model, hyperpol_amp, delay, duration, "soma", 0.5, "soma", 0.5)

        pool = multiprocessing.Pool(1, maxtasksperchild = 1)
        traces = pool.apply(self.cclamp, args = (model, amplitude, hyperpol_amp, delay, duration, "soma", 0.5, dend_locations))


        #plt.close('all') #needed to avoid overlapping of saved images when the test is run on multiple models
        print('Plot traces...')
        self.plot_traces(model, traces, dend_locations, actual_distances)

        traces_results = self.extract_somatic_spiking_features(traces, delay, duration)


        features, AP_amp_soma, soma_ap1, duration_waveform = self.extract_amplitudes(traces, delay, traces_results, actual_distances)
        self.plot_features(model, features, AP_amp_soma, actual_distances,Na_block = False)
        pool2 = multiprocessing.Pool(1, maxtasksperchild = 1)
        traces_vclamp = pool2.apply(self.waveform_vclamp, args = (model, soma_ap1, delay, duration_waveform, "soma", 0.5, dend_locations))

        filepath = self.path_results + self.test_log_filename
        self.logFile = open(filepath, 'w') # if it is opened before multiprocessing, the multiporeccing won't work under python3

        self.logFile.write('Dendritic locations to be tested (with their actual distances):\n'+ str(actual_distances)+'\n')
        self.logFile.write("---------------------------------------------------------------------------------------------------\n")

        self.logFile.write(message_to_logFile)

        self.plot_traces(model, traces_vclamp, dend_locations, actual_distances)
        features2, AP_amp_soma, soma_ap1, duration_waveform = self.extract_amplitudes(traces_vclamp, delay, traces_results, actual_distances)
        #print(features, features2)
        self.plot_features(model, features2, AP_amp_soma, actual_distances,Na_block = True)

        combined_features = {}
        combined_features['active_Na'] = features
        combined_features['blocked_Na'] = features2 

        features_json = collections.OrderedDict()
        for key1 in combined_features:
            features_json[key1] = collections.OrderedDict()
            for key in combined_features[key1]:
                features_json[key1][key] = collections.OrderedDict()
                for ke in combined_features[key1][key]:
                    features_json[key1][key][str(ke)] = collections.OrderedDict()
                    for k, value in combined_features[key1][key][ke].items():
                        features_json[key1][key][str(ke)][k] = str(value)
                
       # print('Features_json: ',features_json)

        # generating prediction
        for key1 in combined_features:
            prediction[key1] = {}  
            for key in combined_features[key1]:  
                AP1_amps = numpy.array([])
           # APlast_amps = numpy.array([])

                for k in combined_features[key1][key]:
                    AP1_amps = numpy.append(AP1_amps, combined_features[key1][key][k]['AP1_amp']/AP_amp_soma)#itt osztottam 
                prediction[key1] = {'model_AP1_amp_dendpersoma_at_'+str(key)+'um':{}}
                prediction[key1]['model_AP1_amp_dendpersoma_at_'+str(key)+'um']['mean'] = float(numpy.mean(AP1_amps))*mV
                prediction[key1]['model_AP1_amp_dendpersoma_at_'+str(key)+'um']['std'] = float(numpy.std(AP1_amps))*mV

        print('Prediction... ',prediction)
        prediction_json = collections.OrderedDict()
        for key1 in prediction:
            prediction_json[key1] = collections.OrderedDict()
            for key in prediction[key1]:
                prediction_json[key1][key] = collections.OrderedDict()
                for k, value in prediction[key1][key].items():
                    prediction_json[key1][key][k]=str(value)


        file_name_json = self.path_results + 'bAP_model_features_means.json'
        json.dump(prediction_json, open(file_name_json, "w"), indent=4)
        file_name_features_json = self.path_results + 'bAP_model_features.json'
        json.dump(features_json, open(file_name_features_json, "w"), indent=4)

        if self.save_all:
            file_name_pickle = self.path_results + 'bAP_model_features.p'

            pickle.dump(features, gzip.GzipFile(file_name_pickle, "wb"))

            file_name_pickle = self.path_results + 'bAP_model_features_means.p'

            pickle.dump(prediction, gzip.GzipFile(file_name_pickle, "wb"))
        print('Plot features...')
       # self.plot_features(model, features, AP_amp_soma, actual_distances)

        efel.reset()

        return prediction

    def compute_score(self, observation, prediction, verbose=False):
        """Implementation of sciunit.Test.score_prediction."""
        print('Compute score')
        print('Observation... ',observation)
        distances = numpy.array(self.config['recording']['distances'])
        #score_sum_StrongProp, score_sum_WeakProp  = scores.ZScore_backpropagatingAP.compute(observation,prediction, [50, 150, 250])
        score_avg, errors= scores.ZScore_backpropagatingAP_OLM.compute(observation,prediction, distances)
        print(10)
        scores_dict = {}
        scores_dict['Z_score_avg'] = score_avg
       # scores_dict['Z_score_avg_weak_propagating'] = score_avg[1]

        file_name=self.path_results+'bAP_errors.json'

        json.dump(errors, open(file_name, "w"), indent=4)

        file_name_s=self.path_results+'bAP_scores.json'

        json.dump(scores_dict, open(file_name_s, "w"), indent=4)
        print('Errors: ',errors)
        self.plot_results(observation, prediction, errors, model_name_bAP)

        if self.show_plot:
            plt.show()

        best_score = score_avg
        score_json = {'Z_score_avg' : best_score}  

        file_name_score = self.path_results + 'bAP_final_score.json'
        json.dump(score_json, open(file_name_score, "w"), indent=4)


        score=scores.ZScore_backpropagatingAP_OLM(best_score)

        self.logFile.write(str(score)+'\n')
        self.logFile.write("---------------------------------------------------------------------------------------------------\n")


        self.logFile.close()

        self.logFile = self.path_results + self.test_log_filename

        return score

    def bind_score(self, score, model, observation, prediction):

        score.related_data["figures"] = [self.path_figs + 'AP1_amp_means.pdf', self.path_figs + 'AP1_amps.pdf', self.path_figs + 'AP1_traces.pdf',
                                       # self.path_figs + 'APlast_amp_means.pdf', self.path_figs + 'APlast_amps.pdf', self.path_figs + 'APlast_traces.pdf', self.path_figs + 'Spikecounts_bAP.pdf',
                                        self.path_figs + 'bAP_errors.pdf', self.path_figs + 'traces.pdf', self.path_results + 'bAP_errors.json',
                                        self.path_results + 'bAP_model_features.json', self.path_results + 'bAP_model_features_means.json',
                                        self.path_results + 'bAP_scores.json', self.path_results + 'bAP_final_score.json', self.path_results + self.test_log_filename]
        score.related_data["results"] = [self.path_results + 'bAP_errors.json', self.path_results + 'bAP_model_features.json', self.path_results + 'bAP_model_features_means.json', self.path_results + 'bAP_scores.json', self.path_results + 'bAP_model_features.p', self.path_results + 'bAP_model_features_means.p', self.path_results + 'bAP_final_score.json']
        return score
