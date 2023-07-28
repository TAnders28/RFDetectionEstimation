from logging import Filter
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


##############################
##############################
# Helpers
def butterLowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


# Filter data that is sampled at fs using a low pass butterworth filter of order filt_order
def butterLowpassFilter(data, cutoff, fs, filt_order=5):
    b, a = butterLowpass(cutoff, fs, order=filt_order)
    y = lfilter(b, a, data)
    return y


def normalize(Signal):
    return Signal / np.max(np.abs(Signal))


def scaleToDesiredPower(Signal, DesiredPower=1):
    InitialPower = np.linalg.norm(Signal) ** 2

    scalingFactor = np.sqrt(DesiredPower / InitialPower)

    Signal = Signal * scalingFactor

    return Signal


##############################
##############################
# Base waveforms
def genSineWave(fc=75000, fs=2048000, Ns=500, A=1, phaseOptional=0, plotDebug=False):
    """
    Generates a sine wave with the specified parameters:
    carrier frequency fc,
    sampling frequency fs,
    Number of samples Ns

    Optional debug flag plotDebug
    """

    isComplex = type(A) == complex
    t = r_[0:Ns] / fs  # time points

    if isComplex == False:
        SineWave = A * np.sin(2 * np.pi * fc * t + phaseOptional)

    else:
        SineWave = A * np.exp(1j * (2 * np.pi * fc * t))

    if plotDebug:
        plt.plot(SineWave)
        plt.show()

    return SineWave


def genBPSK(
    A=1,
    fc=75000,
    fs=2048000,
    Ns=500,
    SymbolRate=25000,
    printDebug=False,
    plotDebug=False,
):
    """
    Generate samples of BPSK with the specified parameters:
    carrier frequency fc,
    sampling frequency fs,
    Number of samples Ns,
    Symbol rate Symbolrate

    Optional debug flags printDebug and plotDebug
    """
    NsSymbol = fs / SymbolRate
    NbitsBPSK = int(np.ceil(Ns / NsSymbol))

    if printDebug:
        print("BPSK samples: ", Ns)

    offset = rm.randint(0, 1000)
    t = r_[0.0 + offset : Ns + offset] / fs  # time points
    inputBits = np.asarray(np.random.randn(NbitsBPSK, 1) > 0)
    carrier = A * cos(2 * pi * fc * t)

    if printDebug:
        print(I_bits)

    start = 0

    I_signal = []
    I_index = 0
    I_bits = inputBits * 2 - 1
    NextSymbolTime = NsSymbol

    while start < Ns:
        I_signal.append(I_bits[I_index])
        start = start + 1
        if start > NextSymbolTime:
            I_index = I_index + 1
            NextSymbolTime = NextSymbolTime + NsSymbol

    I_signal = np.array(I_signal).ravel()

    rrc = komm.RootRaisedCosinePulse(0.4, 10)
    symbol_array = np.arange(-10.5, 10.5, 1 / NsSymbol)
    if plotDebug:
        plt.plot(symbol_array)
        plt.show()
    rrc_response = np.empty([len(symbol_array)])
    o = 0
    for i in symbol_array:
        rrc_response[o] = rrc.impulse_response(i)
        o = o + 1
    if plotDebug:
        plt.plot(rrc_response)
        plt.show()
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
        axis[0].plot(t, BPSK_Signal, color="C1")
        axis[1].plot(t, I_signal, color="C2")
        plt.subplots_adjust(hspace=0.5)
        plt.show()
    return BPSK_Signal


def genQPSK(
    fc=75000, fs=2048000, Ns=500, SymbolRate=25000, printDebug=False, plotDebug=False
):
    """
    Generate samples of QPSK with the specified parameters:
    carrier frequency fc,
    sampling frequency fs,
    Number of samples Ns,
    Symbol rate Symbolrate

    Optional debug flags printDebug and plotDebug
    """
    NsSymbol = fs / SymbolRate
    NbitsQPSK = int(np.ceil(Ns / NsSymbol)) * 2

    offset = rm.randint(0, 1000)
    t = r_[0.0 + offset : Ns + offset] / fs  # time points
    if printDebug:
        print("QPSK samples: ", Ns)
    # Input of the modulator
    inputBits = np.random.randn(NbitsQPSK, 1) > 0

    # Carrier signals used for modulation.
    carrier1 = cos(2 * pi * fc * t)
    carrier2 = sin(2 * pi * fc * t)

    I_bits = inputBits[::2]
    Q_bits = inputBits[1::2]

    start = 0

    I_signal = []
    I_index = 0
    NextSymbolTime = NsSymbol

    I_bits = I_bits * 2 - 1

    while start < Ns:
        I_signal.append(I_bits[I_index])
        start = start + 1
        if start > NextSymbolTime:
            I_index = I_index + 1
            NextSymbolTime = NextSymbolTime + NsSymbol

    I_signal = np.array(I_signal).ravel()

    start = 0
    Q_signal = []
    Q_index = 0
    NextSymbolTime = NsSymbol

    Q_bits = Q_bits * 2 - 1

    while start < Ns:
        Q_signal.append(Q_bits[Q_index])
        start = start + 1
        if start > NextSymbolTime:
            Q_index = Q_index + 1
            NextSymbolTime = NextSymbolTime + NsSymbol

    Q_signal = np.array(Q_signal).ravel()

    rrc = komm.RootRaisedCosinePulse(0.4, 10)
    symbol_array = np.arange(-10.5, 10.5, 1 / NsSymbol)
    rrc_response = np.empty([len(symbol_array)])

    if plotDebug:
        plt.plot(symbol_array)
        plt.show()

    o = 0
    for i in symbol_array:
        # print(o)
        # print(rrc.impulse_response(i))
        rrc_response[o] = rrc.impulse_response(i)
        o = o + 1

    if plotDebug:
        plt.plot(rrc_response)
        plt.show()

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
    if plotDebug:
        plt.plot(I_signal_pulse)
        plt.plot(Q_signal_pulse)
        plt.show()

    I_signal_modulated = I_signal_pulse * carrier1
    Q_signal_modulated = Q_signal_pulse * carrier2

    QPSK_signal = I_signal_modulated + Q_signal_modulated

    QPSK_signal = QPSK_signal / max(abs(QPSK_signal))

    if plotDebug:
        fig, axis = plt.subplots(3, 1, sharex="col")
        fig.suptitle("QPSK Signal", fontsize=12)
        axis[0].plot(t, QPSK_signal, color="C1")
        axis[1].plot(t, I_signal, color="C2")
        axis[2].plot(t, Q_signal, color="C3")
        plt.subplots_adjust(hspace=0.5)

        plt.show()
    return QPSK_signal


def genScaledNoise(Signal, SNR, complexOut=False):
    SNRLin = pow(10, SNR / 10)

    SigPower = np.linalg.norm(Signal) ** 2

    NoisePowerDesired = SigPower / SNRLin

    awgn = np.random.normal(0, 1, len(Signal))

    if complexOut:
        awgn = awgn + 1j * np.random.normal(0, 1, len(Signal))
    awgn = scaleToDesiredPower(awgn, NoisePowerDesired)

    return awgn


def genConstantNoise(Ns=500, complexOut=False):
    awgn = np.random.normal(0, np.sqrt(2) / 2, Ns)

    if complexOut:
        awgn = awgn + 1j * np.random.normal(0, np.sqrt(2) / 2, Ns)

    return awgn


##############################
##############################
# Noise added to base waveforms
def genSineWaveNoise(
    SNR, fc=75000, fs=2048000, Ns=500, complexOut=False, scaled="Noise", plotDebug=False
):
    """
    Generates a noisy sine wave with the specified parameters:
    Signal-Noise Ratio SNR,
    carrier frequency fc,
    sampling frequency fs,
    Number of samples Ns

    Optional debug flag plotDebug
    """

    if complexOut:
        A = np.sqrt(2) / 2 + 1j * np.sqrt(2) / 2
    else:
        A = 1

    SineWave = genSineWave(fc, fs, Ns, A=A)
    if scaled == "Noise":
        Noise = genScaledNoise(SineWave, SNR, complexOut=complexOut)

    else:
        Noise = genConstantNoise(Ns=Ns, complexOut=complexOut)

        NoisePower = np.linalg.norm(Noise) ** 2
        SNRLin = pow(10, SNR / 10)
        SignalPowerDesired = SNRLin * NoisePower

        SineWave = scaleToDesiredPower(SineWave, SignalPowerDesired)

    SineWaveNoise = SineWave + Noise
    # print(
    #     "SNR True",
    #     10 * np.log10(np.linalg.norm(SineWave) ** 2 / np.linalg.norm(Noise) ** 2),
    # )
    if plotDebug:
        plt.plot(SineWaveNoise)
        plt.show()
    return SineWaveNoise


def genBPSKNoise(
    SNR,
    fc=75000,
    fs=2048000,
    Ns=500,
    SymbolRate=25000,
    printDebug=False,
    complexOut=False,
    scaled="Noise",
    plotDebug=False,
):
    """
    Generates a noisy BSPK wave with the specified parameters:
    Signal-Noise Ratio SNR,
    carrier frequency fc,
    sampling frequency fs,
    Number of samples Ns,
    Symbol Rate SymbolRate

    Optional debug flags printDebug plotDebug
    """

    if complexOut:
        A = np.sqrt(2) / 2 + 1j * np.sqrt(2) / 2
    else:
        A = 1

    BPSK = genBPSK(A, fc, fs, Ns, SymbolRate)

    if scaled == "Noise":
        Noise = genScaledNoise(BPSK, SNR, complexOut=complexOut)

    else:
        Noise = genConstantNoise(Ns=len(BPSK), complexOut=complexOut)

        NoisePower = np.linalg.norm(Noise) ** 2
        SNRLin = pow(10, SNR / 10)
        SignalPowerDesired = SNRLin * NoisePower

        BPSK = scaleToDesiredPower(BPSK, SignalPowerDesired)

    BPSKNoise = BPSK + Noise
    print(
        "SNR True",
        10 * np.log10(np.linalg.norm(BPSK) ** 2 / np.linalg.norm(Noise) ** 2),
    )

    if plotDebug:
        plt.plot(BPSK)
        plt.plot(BPSKNoise)
        plt.show()

    return BPSKNoise


def genQPSKNoise(
    SNR,
    fc=75000,
    fs=2048000,
    Ns=500,
    SymbolRate=25000,
    printDebug=False,
    plotDebug=False,
):
    """
    Generates a noisy QSPK wave with the specified parameters:
    Signal-Noise Ratio SNR,
    carrier frequency fc,
    sampling frequency fs,
    Number of samples Ns,
    Symbol Rate SymbolRate

    Optional debug flags printDebug plotDebug
    """

    QPSK = genQPSK(fc, fs, Ns, SymbolRate)

    Noise = genScaledNoise(QPSK, SNR)

    QPSKNoise = QPSK + Noise
    print(
        "SNR True",
        10 * np.log10(np.linalg.norm(QPSK) ** 2 / np.linalg.norm(Noise) ** 2),
    )

    if plotDebug:
        # plt.plot(QPSK)
        # plt.plot(QPSK_NoiseNormal)
        plt.plot(QPSK)
        plt.plot(QPSKNoise)
        plt.show()

    return QPSKNoise


##############################
##############################
# Simulated Downconversion
def convertToIQ(
    SignalRecieved, fReciever=75000, fs=2048000, bandwidth=40000, plotDebug=False
):
    """
    Converts 1 channel of raw input into 2 channels
    representing the In-Phase and Quadrature components
    as interpreted by a reciever tuned to frequency fReciever,
    with samples taken at sampling frequency fs
    """
    offset = rm.randint(0, 200)

    t = r_[0.0 + offset : len(SignalRecieved) + offset + 200] / fs  # time points

    appendedPoints = np.random.randn(200)
    appendedPoints /= np.max(np.abs(appendedPoints))
    SignalRecieved = np.append(appendedPoints, SignalRecieved)

    phase_shift = 2 * pi * rm.random()
    FinalSignal_I = SignalRecieved * cos(2 * pi * fReciever * t + phase_shift)
    FinalSignal_Q = SignalRecieved * sin(2 * pi * fReciever * t + phase_shift)

    if plotDebug:
        plt.plot(SignalRecieved)
        plt.plot(FinalSignal_I)
        plt.plot(FinalSignal_Q)
        plt.show()

    FinalSignal_I_Filtered = butterLowpassFilter(FinalSignal_I, bandwidth, fs)
    FinalSignal_Q_Filtered = butterLowpassFilter(FinalSignal_Q, bandwidth, fs)
    if plotDebug:
        plt.plot(SignalRecieved)
        plt.plot(FinalSignal_I_Filtered)
        plt.plot(FinalSignal_Q_Filtered)
        plt.show()

    FinalSignal_I_Filtered = FinalSignal_I_Filtered[300:]
    FinalSignal_Q_Filtered = FinalSignal_Q_Filtered[300:]

    result = FinalSignal_I_Filtered + 1j * FinalSignal_Q_Filtered
    return result


##############################
##############################
# Sine data test file generation after downconversion to complex baseband
def genSineFile(
    SNRs=np.arange(-10, 5),
    fc=75000,
    fc_delta=0,
    fs=2048000,
    NumberOfRecordings=100,
    Ns=500,
    complexOut=False,
    scaled="Noise",
):
    """
    Generates a fully formatted matrix of shape(NumberOfRecordings, Ns, 2),
    containing NumberOfRecordings of length NS of a noisy sine wave with
    the specified parameters.
    """

    Result = np.zeros((len(SNRs), NumberOfRecordings, Ns), dtype=complex)

    for SNR in SNRs:
        for i in range(NumberOfRecordings):
            FrequencyError = 2 * fc_delta * np.random.rand() - fc_delta
            SNRRandRange = SNR - 0.5 + rm.random()
            NewRecording = convertToIQ(
                genSineWaveNoise(
                    SNRRandRange, fc + FrequencyError, fs, Ns + 100, complexOut, scaled
                ),
                fc,
                fs,
            )
            print(NewRecording.shape)
            Result[SNR - (np.min(SNRs)), i, :] = NewRecording

    Result = np.array(Result)
    return Result

##############################
##############################
# Sine data test file generation at RF or an intermediate complex frequency. 
def genSineFileRF(
    SNRs=np.arange(-10, 5),
    fc=75000,
    fc_delta=0,
    fs=2048000,
    NumberOfRecordings=100,
    Ns=500,
    complexOut=False,
    scaled="Noise",
):
    """
    Generates a fully formatted matrix of shape(NumberOfRecordings, Ns, 2),
    containing NumberOfRecordings of length NS of a noisy sine wave with
    the specified parameters.
    """

    if complexOut:
        Result = np.zeros((len(SNRs), NumberOfRecordings, Ns), dtype=complex)
    else:
        Result = np.zeros((len(SNRs), NumberOfRecordings, Ns))

    for SNR in SNRs:
        for i in range(NumberOfRecordings):
            FrequencyError = 2 * fc_delta * np.random.rand() - fc_delta

            SNRRandRange = SNR - 0.5 + rm.random()

            NewRecording = genSineWaveNoise(
                SNRRandRange, fc + FrequencyError, fs, Ns, complexOut, scaled
            )
            # print(NewRecording.shape)
            Result[SNR - (np.min(SNRs)), i, :] = NewRecording

    Result = np.array(Result)
    return Result


def genNoiseFile(fc=75000, fs=2048000, NumberOfRecordings=100, Ns=500):
    """
    Generates a fully formatted matrix of shape(NumberOfRecordings, Ns, 2),
    containing NumberOfRecordings of length NS of 2 channel AWGN with
    the specified parameters.
    """
    result = np.zeros((NumberOfRecordings, Ns), dtype=complex)
    index = 0
    for i in range(NumberOfRecordings):
        newRecording = convertToIQ(genConstantNoise(Ns=Ns + 100), fc, fs)

        result[index, :] = newRecording

        index += 1

    result = np.array(result)
    print(result.shape)
    return result


def genNoiseFileRF(
    fc=75000, fs=2048000, NumberOfRecordings=100, Ns=500, complexOut=False
):
    """
    Generates a fully formatted matrix of shape(NumberOfRecordings, Ns, 2),
    containing NumberOfRecordings of length NS of 2 channel AWGN with
    the specified parameters.
    """
    result = np.zeros((NumberOfRecordings, Ns), dtype=complex)
    index = 0
    for i in range(NumberOfRecordings):
        newRecording = genConstantNoise(Ns=Ns, complexOut=complexOut)

        result[index, :] = newRecording

        index += 1

    result = np.array(result)
    print(result.shape)
    return result

##############################
##############################
# QPSK data file generation
def genQPSKFile(
    SNRs=np.arange(-10, 5),
    fc=75000,
    fc_delta=0,
    fs=2048000,
    SymbolRate=25000,
    NumberOfRecordings=100,
    Ns=500,
):
    """
    Generates a fully formatted matrix of shape(NumberOfRecordings, Ns, 2),
    containing NumberOfRecordings of length NS of a noisy QPSK signal with
    the specified parameters.
    """
    Result = np.zeros((len(SNRs), NumberOfRecordings, Ns), dtype=complex)

    for SNR in SNRs:
        for i in range(NumberOfRecordings):
            FrequencyError = 2 * fc_delta * np.random.rand() - fc_delta
            SNRRandRange = SNR - 0.5 + rm.random()
            NewRecording = convertToIQ(
                genQPSKNoise(
                    SNRRandRange, fc + FrequencyError, fs, Ns + 100, SymbolRate
                ),
                fc,
                fs,
            )
            Result[SNR - (np.min(SNRs)), i, :] = NewRecording
    Result = np.array(Result)
    return Result

##############################
##############################
# BPSK data file generation
def genBPSKFile(
    SNRs=np.arange(-10, 5),
    fc=75000,
    fc_delta=0,
    fs=2048000,
    SymbolRate=25000,
    NumberOfRecordings=100,
    Ns=500
):
    """
    Generates a fully formatted matrix of shape(NumberOfRecordings, Ns, 2),
    containing NumberOfRecordings of length NS of a noisy BPSK signal with
    the specified parameters.
    """

    Result = np.zeros((len(SNRs), NumberOfRecordings, Ns), dtype=complex)

    for SNR in SNRs:
        for i in range(NumberOfRecordings):
            FrequencyError = 2 * fc_delta * np.random.rand() - fc_delta
            SNRRandRange = SNR - 0.5 + rm.random()
            NewRecording = convertToIQ(
                genBPSKNoise(
                    SNRRandRange, fc + FrequencyError, fs, Ns + 100, SymbolRate
                ),
                fc,
                fs,
            )
            Result[SNR - (np.min(SNRs)), i, :] = NewRecording
    Result = np.array(Result)
    return Result


def genPairedSineWaveParamsFile(
    SNRs=np.arange(-5, 15), NumberOfRecordings=100, fc=75000, fc_delta = 0, fs=2048000, Ns=500
):
    Params = np.zeros((len(SNRs),NumberOfRecordings, 3))
    Noisy = np.zeros((len(SNRs),NumberOfRecordings, Ns), dtype=complex)
    for SNR in SNRs:
        for i in np.arange(NumberOfRecordings):
            SNRRandRange = SNR - 0.5 + rm.random()
            FrequencyError = 2 * fc_delta * np.random.rand() - fc_delta
            SNRLin = pow(10, SNRRandRange / 10)
            noise = genConstantNoise(Ns=Ns)
            NoisePower = np.sum(np.abs(noise)** 2) / Ns
            SignalPower = NoisePower * SNRLin

            waveOG = genSineWave(A = 1 + 1j, fc=fc + FrequencyError, fs=fs, Ns=Ns, plotDebug=False)
            waveScaled = scaleToDesiredPower(waveOG, SignalPower)
            SignalPower = np.sum(np.abs(waveOG)** 2) / Ns

            # print("Signal Power:",SignalPower)
            # print("Noise Power:",NoisePower)
            Params[SNR - (np.min(SNRs)), i,:] = [fc + FrequencyError, np.real(waveScaled[0]), np.imag(waveScaled[0])]
            waveNoisy = waveOG + noise
            Noisy[SNR - (np.min(SNRs)), i,:] = waveNoisy
    return Noisy, Params