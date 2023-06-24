import numpy as np

def EstimateAmplitude(input):
    N = np.size(input)
    
    
    M = max(N,100)
    i = 0 

    x = np.zeros_like(input,dtype=complex)
 
    indexes = np.arange(N) 
    while i < M:
 
        total = np.sum(input[indexes] * np.exp(-1j * i * indexes/(M-1) ))

        x[i] = (1 / N) * total
        
        i = i + 1

    i_max = np.argmax(np.power(np.abs(x),2))

    estimate = x[i_max]

    return estimate

def EstimateFrequency(input):
    estimate = 0
    N= len(input)
    M = N

    # input = input / np.max(np.abs(input))

    p = np.zeros_like(np.arange(M),dtype=float)

    for m in range(M-1):
        p[m] = (6 * (m+1) * (M-(m+1))) / (M * (M**2 - 1))
        newTerm = p[m] * np.angle(np.conj(input[m]) * input[m+1], deg=False)
        estimate = estimate + newTerm
        # print(estimate)

    return estimate


def CRLBFrequency(SNR,N):

    MSE=6/(SNR*N*(N**2-1))
    print(MSE)
    return MSE

def CRLBAmplitude(NoiseVariance,N):

    MSE=NoiseVariance**2/N
    return MSE

