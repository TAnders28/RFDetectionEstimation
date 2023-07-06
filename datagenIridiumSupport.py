import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy as scy
import komm
from tokenize import Number
from numpy import pi
from numpy import sin
from numpy import cos
from numpy import r_
from numpy import random as rm
from scipy.signal import butter, lfilter
from scipy.fftpack import fft
from logging import Filter
import datagen




##############################
##############################
# Iridium Signal simulation
def genIridiumSimplex(
    SNRIn, fc=75000, fs=2048000, RecordingLen=500, printDebug=False, plotDebug=False
):
    """
    Generates a segmented Iridium Simplex Signal with the specified parameters, including
    carrier frequency fc,
    sampling frequency fs,
    Recording length RecordingLen
    Additionally, includes optional debug options
    """

    bits = 50000  # Bit rate
    nBits = 1016  # Total number of bits
    f0 = fc

    N = int(nBits * fs / bits)  # Total Number of Samples
    if printDebug:
        print(N)
    padding = int(RecordingLen - (N % RecordingLen))  # Padding to ensure even splits
    N = N + padding
    if printDebug:
        print("Total samples: ", N)
    offset = rm.randint(0, 1000)

    t = r_[0.0 + offset : N + offset] / fs  # time points

    if printDebug:
        print("Total Timesteps: ", len(t))

    ############################
    # NOISE BEFORE SIGNAL
    preSignalNoiseLen = rm.randint(0, padding) + 100
    preSignalNoise = np.zeros(preSignalNoiseLen)
    if printDebug:
        print("Pre signal noise samples: ", preSignalNoiseLen)

    ############################
    # PREAMBLE

    symbs = 25000
    Ns = fs / (symbs)
    preambleStartIndex = preSignalNoiseLen
    NBitsPreamble = 64
    NPreamble = int(NBitsPreamble * Ns)
    preambleEndIndex = preambleStartIndex + NPreamble
    tPreamble = t[preambleStartIndex:preambleEndIndex]

    if printDebug:
        print("Preamble samples: ", NPreamble)

    phase_shift = 2 * pi * rm.random()
    carrier = sin(2 * pi * f0 * tPreamble + phase_shift)
    Preamble_Signal = carrier

    if plotDebug:
        plt.plot(Preamble_Signal)
        plt.show()
    ############################
    # BPSK

    NbitsBPSK = 12
    BPSKStartIndex = preambleEndIndex
    BPSKSymbolRate = 25000
    Ns = fs / BPSKSymbolRate
    NBPSK = int(NbitsBPSK * Ns)
    BPSKEndIndex = BPSKStartIndex + int(NBPSK)
    # printDebug=True
    if printDebug:
        print("BPSK samples: ", NBPSK)
    tBPSK = t[BPSKStartIndex:BPSKEndIndex]
    carrier = sin(2 * pi * f0 * tBPSK + phase_shift)

    inputBits = [[0], [1], [1], [1], [1], [0], [0], [0], [1], [0], [0], [1]]
    I_bits = np.asarray(inputBits)

    start = 0

    I_signal = np.zeros((NBPSK, 1))
    I_index = 0
    NextSymbolTime = Ns

    I_bits = I_bits * 2 - 1

    # Interpolating Filter
    while start < NBPSK:
        I_signal[start] = I_bits[I_index]
        start = start + 1
        if start > NextSymbolTime:
            I_index = I_index + 1
            NextSymbolTime = NextSymbolTime + Ns
    I_signal = np.array(I_signal).ravel()
    if plotDebug:
        plt.plot(I_signal)
        plt.show()

    rrc = komm.RootRaisedCosinePulse(0.4, 10)

    symbol_array = np.arange(-10.5, 10.5, 1 / Ns)
    rrc_response = np.empty([len(symbol_array)])
    o = 0
    for i in symbol_array:
        rrc_response[o] = rrc.impulse_response(i)
        o = o + 1

    I_signal_pulse = np.convolve(I_signal, rrc_response)
    I_signal_pulse = I_signal_pulse[
        int(len(rrc_response) / 2) : len(I_signal_pulse)
        - int(len(rrc_response) / 2)
        + (len(rrc_response) + 1) % 2
    ]
    if plotDebug:
        plt.plot(I_signal_pulse)
        plt.show()
    BPSK_Signal = I_signal_pulse * carrier
    BPSK_Signal = BPSK_Signal / max(abs(BPSK_Signal))
    if plotDebug:
        plt.plot(BPSK_Signal)
        plt.show()

    if plotDebug:
        fig, axis = plt.subplots(2, 1, sharex="col")
        fig.suptitle("BPSK Signal", fontsize=12)
        axis[0].plot(tBPSK, BPSK_Signal, color="C1")
        axis[1].plot(tBPSK, I_signal, color="C2")
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    ############################
    # QPSK

    NbitsQPSK = 864  # number of bits
    QPSKStartIndex = BPSKEndIndex
    Ns = fs / 25000
    NQPSK = int((NbitsQPSK / 2) * Ns)  # Total Number of Samples
    NQPSK = NQPSK + NQPSK % 2
    # printDebug=True
    if printDebug:
        print("QPSK samples: ", NQPSK)
    QPSKEndIndex = QPSKStartIndex + NQPSK

    tQPSK = t[QPSKStartIndex:QPSKEndIndex]  # time points

    # Input of the modulator
    inputBits = np.random.randn(NbitsQPSK, 1) > 0

    # Carrier signals used for modulation.
    carrier1 = cos(2 * pi * f0 * tQPSK + phase_shift + np.pi / 4)
    carrier2 = sin(2 * pi * f0 * tQPSK + phase_shift + np.pi / 4)
    if printDebug:
        print(carrier1.shape)

    I_bits = inputBits[::2]
    Q_bits = inputBits[1::2]

    start = 0
    I_signal = np.zeros((NQPSK, 1))
    I_index = 0
    NextSymbolTime = Ns

    I_bits = I_bits * 2 - 1

    while start < NQPSK:
        I_signal[start] = I_bits[I_index]
        start = start + 1
        if start > NextSymbolTime:
            I_index = I_index + 1
            NextSymbolTime = NextSymbolTime + Ns

    I_signal = np.array(I_signal).ravel()

    start = 0
    Q_signal = np.zeros((NQPSK, 1))
    Q_index = 0
    NextSymbolTime = Ns

    Q_bits = Q_bits * 2 - 1

    while start < NQPSK:
        Q_signal[start] = Q_bits[Q_index]
        start = start + 1
        if start > NextSymbolTime:
            Q_index = Q_index + 1
            NextSymbolTime = NextSymbolTime + Ns

    Q_signal = np.array(Q_signal).ravel()

    rrc = komm.RootRaisedCosinePulse(0.4, 10)
    symbol_array = np.arange(-10.5, 10.5, 1 / Ns)
    rrc_response = np.empty([len(symbol_array)])
    o = 0
    for i in symbol_array:
        # print(o)
        # print(rrc.impulse_response(i))
        rrc_response[o] = rrc.impulse_response(i)
        o = o + 1

    I_signal_pulse = np.convolve(I_signal, rrc_response)
    Q_signal_pulse = np.convolve(Q_signal, rrc_response)

    I_signal_pulse = I_signal_pulse[
        int(len(rrc_response) / 2) : len(I_signal_pulse)
        - int(len(rrc_response) / 2)
        + (len(rrc_response) + 1) % 2
    ]
    Q_signal_pulse = Q_signal_pulse[
        int(len(rrc_response) / 2) : len(Q_signal_pulse)
        - int(len(rrc_response) / 2)
        + (len(rrc_response) + 1) % 2
    ]
    I_signal_modulated = I_signal_pulse * carrier1
    Q_signal_modulated = Q_signal_pulse * carrier2

    # Summation before transmission
    QPSK_signal = I_signal_modulated + Q_signal_modulated

    # Normalize
    QPSK_signal = QPSK_signal / max(abs(QPSK_signal))

    if plotDebug:
        fig, axis = plt.subplots(3, 1, sharex="col")
        fig.suptitle("QPSK Signal", fontsize=12)
        axis[0].plot(tQPSK, QPSK_signal, color="C1")
        axis[1].plot(tQPSK, I_signal, color="C2")
        axis[2].plot(tQPSK, Q_signal, color="C3")
        plt.subplots_adjust(hspace=0.5)

        plt.show()

    ############################
    # NOISE AFTER SIGNAL

    postSignalNoiseLen = padding - preSignalNoiseLen

    postSignalNoise = np.zeros(postSignalNoiseLen)

    if printDebug:
        print("Post signal noise samples: ", postSignalNoiseLen)

    BPSK_Signal = BPSK_Signal / max(abs(BPSK_Signal))
    QPSK_Signal = QPSK_signal / max(abs(QPSK_signal))
    Preamble_Signal = Preamble_Signal / max(abs(Preamble_Signal))
    Pre_Mod_Signal = np.append(BPSK_Signal, QPSK_Signal)
    FinalSignal_TX = np.append(Preamble_Signal, Pre_Mod_Signal)
    FinalSignal_TX = np.append(preSignalNoise, FinalSignal_TX)
    FinalSignal_TX = np.append(FinalSignal_TX, postSignalNoise)

    Ts = 1 / fs
    if plotDebug:
        bitsToShow = 10000
        timeDomainVisibleLimit = Ts * len(FinalSignal_TX)
        fig, axis = plt.subplots(3, 1, sharex="col")
        fig.suptitle("QPSK Modulation", fontsize=12)
        axis[0].plot(t, np.real(FinalSignal_TX), color="C1")
        axis[0].set_xlim(0, timeDomainVisibleLimit)
        axis[0].set_ylabel("Amplitude")
        axis[0].grid(linestyle="dotted")

        # axis[1].plot(t,np.imag(FinalSignal_TX), color='C2')
        # axis[1].set_xlim(0,timeDomainVisibleLimit)
        # axis[1].set_ylabel('Amplitude')
        # axis[1].grid(linestyle='dotted')

        axis[1].plot(t, np.abs(FinalSignal_TX), color="C3")
        axis[1].set_xlim(0, timeDomainVisibleLimit)
        axis[1].set_ylabel("Abs")
        axis[1].grid(linestyle="dotted")

        AngleTemp = np.angle(FinalSignal_TX)

        AngleTemp = (2 * np.pi + AngleTemp) * (AngleTemp < 0) + AngleTemp * (
            AngleTemp > 0
        )
        axis[2].plot(t, AngleTemp, color="C4")
        axis[2].set_xlabel("Time [s]")
        axis[2].set_xlim(0, timeDomainVisibleLimit)
        axis[2].set_ylabel("Angle")
        axis[2].set_ylim(0, 2 * np.pi)
        axis[2].grid(linestyle="dotted")

        plt.subplots_adjust(hspace=0.5)
        plt.show()

    return FinalSignal_TX

    # plotDebug=True
    if plotDebug:
        plt.plot(FinalSignal_TX)
        plt.show()
    # plotDebug=False
    SNRi = SNRIn - 0.5 + rm.random()
    SNRLin = pow(10, SNRi / 10)
    SigPower = np.linalg.norm(FinalSignal_TX) ** 2 / FinalSignal_TX.size
    awgn = komm.AWGNChannel(SNRLin, SigPower)
    FinalSignal_Noise = awgn(FinalSignal_TX)
    FinalSignal_Noise = FinalSignal_Noise / max(abs(FinalSignal_Noise))

    # print(FinalSignal_Noise.shape)
    # plt.plot(FinalSignal_Noise)
    # plt.show()
    Ts = 1 / fs
    t = np.arange(0, Ts * len(FinalSignal_TX), Ts)

    FinalSignal_Noise = np.append(np.random.randn(100), FinalSignal_Noise)

    FinalSignal_I_Filtered, FinalSignal_Q_Filtered = convertToIQ(
        FinalSignal_Noise, f0, fs
    )

    FinalSignal_RX = FinalSignal_I_Filtered + 1j * FinalSignal_Q_Filtered

    # plotDebug=True
    if plotDebug:
        bitsToShow = 10000
        timeDomainVisibleLimit = Ts * len(FinalSignal_RX)
        fig, axis = plt.subplots(4, 1, sharex="col")
        fig.suptitle("QPSK Modulation", fontsize=12)
        axis[0].plot(t, np.real(FinalSignal_RX), color="C1")
        axis[0].set_xlim(0, timeDomainVisibleLimit)
        axis[0].set_ylabel("Amplitude")
        axis[0].grid(linestyle="dotted")

        axis[1].plot(t, np.imag(FinalSignal_RX), color="C2")
        axis[1].set_xlim(0, timeDomainVisibleLimit)
        axis[1].set_ylabel("Amplitude")
        axis[1].grid(linestyle="dotted")

        axis[2].plot(t, np.abs(FinalSignal_RX), color="C3")
        axis[2].set_xlim(0, timeDomainVisibleLimit)
        axis[2].set_ylabel("Abs")
        axis[2].grid(linestyle="dotted")

        AngleTemp = np.angle(FinalSignal_RX)

        AngleTemp = (2 * np.pi + AngleTemp) * (AngleTemp < 0) + AngleTemp * (
            AngleTemp > 0
        )
        axis[3].plot(t, AngleTemp, color="C4")
        axis[3].set_xlabel("Time [s]")
        axis[3].set_xlim(0, timeDomainVisibleLimit)
        axis[3].set_ylabel("Angle")
        axis[3].set_ylim(0, 2 * np.pi)
        axis[3].grid(linestyle="dotted")

        plt.subplots_adjust(hspace=0.5)
        plt.show()
    # plotDebug=False
    fftDebug = False
    # PLOT FFT
    if fftDebug:
        # SegmentStart = random.randint(0, len(FinalSignal_Noise)-500)
        SegmentStart = BPSKStartIndex
        SegmentEnd = SegmentStart + 500
        FFT_Segment = FinalSignal_Noise[SegmentStart:SegmentEnd]
        plt.plot(FFT_Segment)
        plt.show()
        # print(len(FFT_Segment))
        newVec = np.arange(0, fs - fs / len(FFT_Segment) + 1, fs / len(FFT_Segment))
        fft = abs(np.fft.fftshift(np.fft.fft(FFT_Segment)))
        # plt.plot(newVec, fft)
        # plt.show()
        print(max(fft))

    SignalR = np.real(FinalSignal_RX)
    SignalI = np.imag(FinalSignal_RX)
    Signal = np.append([SignalR], [SignalI], axis=0)

    Result = Signal

    if RecordingLen != Result.shape[1]:
        # print(Result.shape)
        ResultSplit = np.asarray(
            np.array_split(Result, int(Result.shape[1] / RecordingLen), axis=1)
        )
        # plt.plot(ResultSplit[0])
        # plt.show()

        ResultSplit = ResultSplit.reshape(
            (ResultSplit.shape[0], ResultSplit.shape[2], ResultSplit.shape[1])
        )
        # plt.plot(ResultSplit[0])
        # plt.show()
        if printDebug:
            print("Output shape: ", ResultSplit.shape)

        # plt.plot(ResultSplit[0])
        # plt.show()
        return ResultSplit
    else:
        # print("Full Signal")
        return Result


##############################
##############################
# Iridium data file generation
def genIridiumFile(SNR, fc=75000, fs=2048000, NumberOfRecordings=100, Ns=500):
    """
    Generates a fully formatted matrix of shape(NumberOfRecordings, Ns, 2),
    containing NumberOfRecordings of length NS of a noisy Iridium signal with
    the specified parameters. Depending on NS, this signal will be sliced in
    pseudorandom places in order to create a robust dataset geared towards machine learning.
    """

    result = []

    newRecording = genIridiumSimplex(SNR, fc, fs, RecordingLen=Ns)
    print(NumberOfRecordings)
    print(newRecording.shape)
    if newRecording.ndim == 2:
        ratio = NumberOfRecordings
    else:
        ratio = np.ceil(NumberOfRecordings / newRecording.shape[0])

    # print("Ratio:", ratio)
    for i in range(int(ratio)):
        newRecording = genIridiumSimplex(SNR, fc, fs, RecordingLen=Ns)
        # print(newRecording.shape)
        if newRecording.ndim == 2:
            newRecording = np.swapaxes(newRecording, 0, 1)
        else:
            newRecording = newRecording

        # print(newRecording.shape)
        result.append(newRecording)

    result = np.array(result)
    if newRecording.ndim != 2:
        result = np.reshape(
            result,
            (result.shape[0] * result.shape[1], result.shape[2], result.shape[3]),
        )
    # print(result.shape)
    result = result[0:NumberOfRecordings, :, :]
    print(result.shape)
    return result


##############################
##############################
# Matched Filter Support


def genMatchedFilter(
    filterLength=500, fc=75000, fs=2048000, printDebug=False, plotDebug=False
):
    """
    Allows support for getting a matched filter of desired length for the Iridium Signal
    """

    bits = 50000  # rate
    nBits = 1016
    f0 = fc

    Ns = fs / bits
    N = int(nBits * fs / bits)  # Total Number of Samples
    if printDebug:
        print(N)
    padding = 0
    N = N + padding
    if printDebug:
        print("Total samples: ", N)
    # offset = rm.randint(0,1000)
    t = r_[0.0:N] / fs  # time points
    if printDebug:
        print("Total Timesteps: ", len(t))

    ############################
    # PREAMBLE

    symbs = 25000
    Ns = fs / (symbs)
    preambleStartIndex = 0
    NBitsPreamble = 64
    NPreamble = int(NBitsPreamble * Ns)
    preambleEndIndex = preambleStartIndex + NPreamble
    tPreamble = t[preambleStartIndex:preambleEndIndex]
    if printDebug:
        print("Preamble samples: ", NPreamble)
    # phase_shift = 2*pi * rm.random()
    phase_shift = 0
    carrier = cos(2 * pi * f0 * tPreamble + phase_shift)

    Preamble_Signal = carrier
    Preamble_Signal = Preamble_Signal / max(abs(Preamble_Signal))

    if plotDebug:
        plt.plot(Preamble_Signal)
        plt.show()
    ############################
    # BPSK

    NbitsBPSK = 12
    BPSKStartIndex = preambleEndIndex
    BPSKSymbolRate = 25000
    Ns = fs / BPSKSymbolRate
    NBPSK = int(NbitsBPSK * Ns)
    BPSKEndIndex = BPSKStartIndex + int(NBPSK)
    if printDebug:
        print("BPSK samples: ", NBPSK)
    tBPSK = t[BPSKStartIndex:BPSKEndIndex]

    inputBits = [[0], [1], [1], [1], [1], [0], [0], [0], [1], [0], [0], [1]]

    carrier = cos(2 * pi * f0 * tBPSK + phase_shift)

    I_bits = np.asarray(inputBits)

    start = 0

    I_signal = []
    I_index = 0
    NextSymbolTime = Ns

    I_bits = I_bits * 2 - 1

    while start < NBPSK:
        I_signal.append(I_bits[I_index])
        start = start + 1
        if start > NextSymbolTime:
            I_index = I_index + 1
            NextSymbolTime = NextSymbolTime + Ns

    I_signal = np.array(I_signal).ravel()

    if plotDebug:
        plt.plot(I_signal)
        plt.show()
    rrc = komm.RootRaisedCosinePulse(0.4, 10)
    symbol_array = np.arange(-10.5, 10.5, 1 / Ns)
    rrc_response = np.empty([len(symbol_array)])
    o = 0
    for i in symbol_array:
        rrc_response[o] = rrc.impulse_response(i)
        o = o + 1
    I_signal_pulse = np.convolve(I_signal, rrc_response)
    I_signal_pulse = I_signal_pulse[
        int(len(rrc_response) / 2) : len(I_signal_pulse)
        - int(len(rrc_response) / 2)
        + (len(rrc_response) + 1) % 2
    ]
    if plotDebug:
        plt.plot(I_signal_pulse)
        plt.show()
    BPSK_Signal = I_signal_pulse * carrier
    BPSK_Signal = BPSK_Signal / max(abs(BPSK_Signal))
    if plotDebug:
        plt.plot(BPSK_Signal)
        plt.show()

    if plotDebug:
        fig, axis = plt.subplots(2, 1, sharex="col")
        fig.suptitle("BPSK Signal", fontsize=12)
        axis[0].plot(tBPSK, BPSK_Signal, color="C1")
        axis[1].plot(tBPSK, I_signal, color="C2")
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    BPSK_Signal = BPSK_Signal / max(abs(BPSK_Signal))
    Preamble_Signal = Preamble_Signal / max(abs(Preamble_Signal))
    result = np.append(Preamble_Signal, BPSK_Signal)
    resultI, resultQ = convertToIQ(result, fc, fs)
    FinalSignal_RX = resultI + 1j * resultQ
    FinalSignal_RX = np.flip(FinalSignal_RX)
    SignalR = np.real(FinalSignal_RX)
    SignalI = np.imag(FinalSignal_RX)
    Signal = np.append([SignalR], [SignalI], axis=0)
    Signal = Signal[:, 0:filterLength]
    print(Signal.shape)
    return Signal


def matchedFilterIridium(SNR, NumberofRecordings, commonThresh=420):
    DataSet = genIridiumFile(SNR, NumberOfRecordings=NumberofRecordings, Ns=42000)
    matchedFilter = genMatchedFilter()
    # print(matchedFilter.shape)
    ComplexMF = matchedFilter[0, :] + 1j * matchedFilter[0, :]
    threshold = commonThresh
    detections = 0
    for signal in DataSet:
        Complex = signal[:, 0] + 1j * signal[:, 1]
        Filtered = np.convolve(ComplexMF, Complex)
        absoluteValFiltered = np.abs(Filtered)
        # plt.plot(absoluteValFiltered)
        if any(absoluteValFiltered > threshold):
            detections = detections + 1


def matchedFilterNoise(NumberofRecordings, commonThresh=420):
    DataSet = genNoiseFile(NumberOfRecordings=NumberofRecordings, Ns=42000)
    matchedFilter = genMatchedFilter()
    # print(matchedFilter.shape)
    ComplexMF = matchedFilter[0, :] + 1j * matchedFilter[0, :]
    threshold = commonThresh
    detections = 0
    for signal in DataSet:
        Complex = signal[:, 0] + 1j * signal[:, 1]
        Filtered = np.convolve(ComplexMF, Complex)
        absoluteValFiltered = np.abs(Filtered)
        # plt.plot(absoluteValFiltered)
        if any(absoluteValFiltered > threshold):
            detections = detections + 1

    # plt.show()

    return detections / NumberofRecordings
