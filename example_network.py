from __future__ import print_function

from brian import second, msecond, ms, Hz, mvolt
import brian
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import collections

# stop appnope from allowing process to sleep
import platform
if platform.mac_ver()[0][:4] == '10.11' or platform.mac_ver()[0][:4] == '10.10':
    import appnope
    appnope.nope()

# histed 160801: extracted from harness that runs on cluster

################

a_ = np.array
r_ = np.r_

# Simulation constants

simRunTimeS = 4.3

taue = 5 * msecond
taui = 10 * msecond
Ee = 0.0 * mvolt
Ei = -80.0 * mvolt
vthresh = -50.0 * mvolt
vrest = -60.0 * mvolt
taum = 20 * msecond
absRefractoryMs = 3

connENetWeight = 0.3
connINetWeight = 9.55
internalSparseness = 0.02
spontAddRate = -4.48  # found via optimization to give mean 5 spk/s spont rate

nNet = 10000
nExc = int(nNet * 0.8)
nInh = int(nNet * 0.2)
condAddNeurNs = r_[0:nExc:4]
ffExcInputNTargs = 2500
ffInhInputNTargs = 666

eqs = brian.Equations('''
dv/dt = (-v+ge*(Ee-v)+gi*(Ei-v)+gAdd*(Ee-v) ) *(1./taum) : volt
dge/dt = -ge*(1./taue) : 1
dgi/dt = -gi*(1./taui) : 1
gAdd : 1
''')

contRecNs = [0]         # record continuous variables (Vm, ge, gi) from one example neuron
contRecStepMs = 1.0


###############

# Functions

def create_input_vectors(doDebugPlot=True):
    """Constructs the feedforward and conductance-add vectors.

    doDebugPlot : boolean, whether or not to plot the timecourses of the vectors, for verification
    Returns:
        ffInputV: vector (dtype f8), size (nTimeptsMs,) - ff input poiss rate timecourse
        condAddV: vector (dtype f8), size (nTimeptsMs,) - added conductance to E cells timecourse
    """
    endMs = int(simRunTimeS * 1000.0)
    ffInputV = np.zeros((endMs,))
    for iT in [1300, 3300]:
        ffInputV[iT:iT + 1000] = 1.0
    condAddV = np.zeros((endMs,))
    condAddV[2300:] = 1.0

    if doDebugPlot:
        plt.figure()
        plt.plot(ffInputV)
        plt.plot(condAddV + 1.2)
        plt.xlabel('time (ms)')
        plt.gca().set_yticklabels('')

    return (ffInputV, condAddV)


def run_sim(ffExcInputMult=None, ffInhInputMult=None):
    """Run the cond-based LIF neuron simulation.  Takes a few minutes to construct network and run


    Parameters
    ----------
    ffExcInputMult: scalar: FF input magnitude to E cells.  multiply ffInputV by this value and connect to E cells
    ffInhInputMult: scalar: FF input magnitude to I cells.

    Returns
    -------
    outDict - spike times, records of continuous values from simulation

    """

    # use helper to get input timecourses
    (ffInputV, condAddV) = create_input_vectors(doDebugPlot=False)  # multiplied by scalars below

    # setup initial state
    stT = time.time()
    brian.set_global_preferences(usecodegen=True)
    brian.set_global_preferences(useweave=True)
    brian.set_global_preferences(usecodegenweave=True)
    brian.clear(erase=True, all=True)
    brian.reinit_default_clock()
    clk = brian.Clock(dt=0.05 * ms)

    ################

    # create neurons, define connections
    neurNetwork = brian.NeuronGroup(nNet, model=eqs, threshold=vthresh,
                              reset=vrest, refractory=absRefractoryMs * msecond,
                              order=1, compile=True, freeze=False, clock=clk)

    # create neuron pools
    neurCE = neurNetwork.subgroup(nExc)
    neurCI = neurNetwork.subgroup(nInh)
    connCE = brian.Connection(neurCE, neurNetwork, 'ge')
    connCI = brian.Connection(neurCI, neurNetwork, 'gi')
    print('n cells: %d, nE,I %d,%d, %s, absRefractoryMs: %d'
          % (nNet, nExc, nInh, repr(clk), absRefractoryMs))

    # connect the network to itself
    connCE.connect_random(neurCE, neurNetwork, internalSparseness, weight=connENetWeight)
    connCI.connect_random(neurCI, neurNetwork, internalSparseness, weight=connINetWeight)

    # connect inputs that change spont rate
    assert (spontAddRate <= 0), 'Spont add rate should be negative - convention: neg, excite inhibitory cells'
    spontAddNInpSyn = 100
    nTotalSpontNeurons = (spontAddNInpSyn * nInh * 0.02)
    neurSpont = brian.PoissonGroup(nTotalSpontNeurons, -1.0 * spontAddRate * Hz)
    connCSpont = brian.Connection(neurSpont, neurCI, 'ge')
    connCSpont.connect_random(p=spontAddNInpSyn * 1.0 / nTotalSpontNeurons,
                              weight=connENetWeight,  # match internal excitatory strengths
                              fixed=True)

    # connect the feedforward visual (poisson) inputs to excitatory cells (ff E)
    ffExcInputNInpSyn = 100   
    nTotalFfNeurons = (ffExcInputNInpSyn * ffExcInputNTargs * 0.02)  # one pop of input cells for both E and I FF
    _ffExcInputV = ffExcInputMult * np.abs(a_(ffInputV).copy())
    assert (np.all(_ffExcInputV >= 0)), 'Negative FF rates are rectified to zero'
    neurFfExcInput = brian.PoissonGroup(nTotalFfNeurons,
                                  lambda t: _ffExcInputV[int(t * 1000)] * Hz)
    connCFfExcInput = brian.Connection(neurFfExcInput, neurNetwork, 'ge')
    connCFfExcInput.connect_random(neurFfExcInput,
                                   neurCE[0:ffExcInputNTargs],
                                   ffExcInputNInpSyn * 1.0 / nTotalFfNeurons,
                                   weight=connENetWeight,
                                   fixed=True)

    # connect the feedforward visual (poisson) inputs to inhibitory cells (ff I)
    ffInhInputNInpSyn = 100
    _ffInhInputV = ffInhInputMult * np.abs(ffInputV.copy())
    assert (np.all(_ffInhInputV >= 0)), 'Negative FF rates are rectified to zero'
    neurFfInhInput = brian.PoissonGroup(nTotalFfNeurons,
                                  lambda t: _ffInhInputV[int(t * 1000)] * Hz)
    connCFfInhInput = brian.Connection(neurFfInhInput, neurNetwork, 'ge')
    connCFfInhInput.connect_random(neurFfInhInput,
                                   neurCI[0:ffInhInputNTargs],
                                   ffInhInputNInpSyn * 1.0 / nTotalFfNeurons,  # sparseness
                                   weight=connENetWeight,
                                   fixed=True)

    # connect added step (ChR2) conductance to excitatory cells
    condAddAmp = 4.0
    gAdd = brian.TimedArray(condAddAmp*condAddV, dt=1 * ms)
    print('Adding conductance for %d cells (can be slow): ' % len(condAddNeurNs), end=' ')
    for (iN, tN) in enumerate(condAddNeurNs):
        neurCE[tN].gAdd = gAdd
    print('done')

    # Initialize using some randomness so all neurons don't start in same state.
    # Alternative: initialize with constant values, give net extra 100-300ms to evolve from initial state.
    neurNetwork.v = (brian.randn(1) * 5.0 - 65) * mvolt
    neurNetwork.ge = brian.randn(nNet) * 1.5 + 4
    neurNetwork.gi = brian.randn(nNet) * 12 + 20

    # Record continuous variables and spikes
    monSTarg = brian.SpikeMonitor(neurNetwork)
    if contRecNs is not None:
        contRecClock = brian.Clock(dt=contRecStepMs * ms)
        monVTarg = brian.StateMonitor(neurNetwork, 'v', record=contRecNs, clock=contRecClock)
        monGETarg = brian.StateMonitor(neurNetwork, 'ge', record=contRecNs, clock=contRecClock)
        monGAddTarg = brian.StateMonitor(neurNetwork, 'gAdd', record=contRecNs, clock=contRecClock)
        monGITarg = brian.StateMonitor(neurNetwork, 'gi', record=contRecNs, clock=contRecClock)

    # construct brian.Network before running (so brian explicitly knows what to update during run)
    netL = [neurNetwork, connCE, connCI, monSTarg,
            neurFfExcInput, connCFfExcInput, neurFfInhInput, connCFfInhInput,
            neurSpont, connCSpont]
    if contRecNs is not None:
        # noinspection PyUnboundLocalVariable
        netL.append([monVTarg, monGETarg, monGAddTarg, monGITarg]) # cont monitors
    net = brian.Network(netL)
    print("Network construction time: %3.1f seconds" % (time.time() - stT))

    # run 
    print("Simulation running...")
    sys.stdout.flush()
    start_time = time.time()
    net.run(simRunTimeS * second, report='text', report_period=30.0 * second)
    durationS = time.time() - start_time
    print("Simulation time: %3.1f seconds" % durationS)

    outNTC = collections.namedtuple('outNTC', 'vm ge gadd gi clockDtS clockStartS clockEndS spiketimes contRecNs')
    outNTC.__new__.__defaults__ = (None,) * len(outNTC._fields)  # default to None
    outNT = outNTC(clockDtS=float(monSTarg.clock.dt),
                   clockStartS=float(monSTarg.clock.start),
                   clockEndS=float(monSTarg.clock.end),
                   spiketimes=a_(monSTarg.spiketimes.values(), dtype='O'),
                   contRecNs=contRecNs)
    if contRecNs is not None:
        outNT = outNT._replace(vm=monVTarg.values,
                               ge=monGETarg.values,
                               gadd=monGAddTarg.values,
                               gi=monGITarg.values)
    return outNT


def hist_spikes(spiketimesA, edgesS, dtype='f8', convert_to_rates=True):
    """Given spike times, returns histogram of spikes for each cell, across time bins
    Parameters:
        spiketimesA: ndarray, dtype='O', as returned by run_sim().spiketimes
        edgesS: vector (f8), histogram edges.  np.r_[0:clockEndS:0.010] gives 10ms bins.  see np.histogram
        dtype: scalar, dtype of output array
        convert_to_rates: boolean, default True.  False means leave as counts in bins
    Returns:
        histM: ndarray (dtype from input), shape (nCells, nBins).

    - nBins is nEdges-1, see np.histogram()
    """

    nCells = len(spiketimesA)
    histM = np.zeros((nCells,len(edgesS)-1), dtype=dtype)
    for iC in range(nCells):
        ctHist = np.histogram(spiketimesA[iC], edgesS)[0].astype(dtype)

        if convert_to_rates:
            binLenS = np.diff(edgesS)
            histM[iC,:] = ctHist*1.0/binLenS  # divide by each bin length; will be broadcast correctly
        else:
            histM[iC,:] = ctHist
    return histM
