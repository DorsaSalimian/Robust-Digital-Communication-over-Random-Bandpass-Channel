# 📡 End-to-End Digital Communication System 

Simulation

This repository contains a complete end-to-end simulation of a digital communication system, implemented in MATLAB, designed to model realistic transmission of text data over an imperfect and randomly varying band-pass channel.

The project was developed as part of an academic Digital Communications course and closely follows practical system-level design used by communication engineers.

🚀 Project Overview

The goal of this project is to transmit a text message through a simulated communication channel and reliably reconstruct it at the receiver under the following impairments:

    Random band removal in the channel frequency response
    Additive white Gaussian noise (AWGN)
    Signal clipping (nonlinear distortion)
    Unknown random time delay (asynchrony)

The system is designed to be robust against these channel effects while maintaining acceptable Bit Error Rate (BER) performance.

🧩 System Architecture

The complete communication chain is implemented:

Text → Binary → Source Encoder → Channel Encoder → Modulator
     → Channel (Random Band Removal + Noise + Clipping)
     → Demodulator (Optional PLL)
     → Detector → Channel Decoder → Source Decoder
     → Binary → Text

📡 Channel Model

The channel is modeled as:

    Causal band-pass filter with four possible frequency bands
    At each transmission, one band is randomly removed
    Band attenuation factor: α
    Noise model:
        Additive White Gaussian Noise (AWGN)
        N(t)∼N(0,σ2)N(t)∼N(0,σ2)
    Signal clipping to simulate hardware nonlinearities
    Unknown random delay between input and output

The channel implementation is provided via a fixed MATLAB function:
simulate_channel_project.m

📤 Transmitter Design

Key transmitter components:

    Text-to-binary conversion
    Source coding (optional Huffman encoding)
    Channel coding for error resilience
    Pulse shaping using a Raised Cosine Filter to limit bandwidth and reduce ISI
    Digital modulation
        Supported schemes: QAM / PSK / FSK
        Configurable modulation order (M)
    Multi-band transmission strategy
        Ensures data recovery even if one band is removed
    Pilot signals
        pilot_start and pilot_end appended to the signal
        Used for synchronization and delay estimation

📥 Receiver Design

At the receiver, the following steps are performed:

    Pilot detection using cross-correlation
    Estimation and removal of random channel delay
    Band analysis
        Detect which frequency band(s) survived the channel
    Demodulation of the available band
    Bit detection under noise
    Channel decoding (Hard-Decision Decoding)
    Source decoding (optional Huffman decoding)
    Binary-to-text reconstruction

Successful reception is verified by comparing the reconstructed text with the original input.

📊 Performance Evaluation

The system performance is evaluated through:

    Time-domain and frequency-domain plots of:
        Channel output
        Signal after pilot removal
        Demodulated baseband signal
    Bit Error Rate (BER) analysis
        BER vs. Data Rate
        BER vs. Noise variance (σ)
    SNR analysis
        Effect of channel attenuation parameter α on SNR
    Comparative analysis:
        With channel coding
        Without channel coding
