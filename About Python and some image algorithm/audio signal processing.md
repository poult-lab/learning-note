# Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between

Apr 21, 2016

Speech processing plays an important role in any speech system whether its Automatic Speech Recognition (ASR) or speaker recognition or something else. Mel-Frequency Cepstral Coefficients (MFCCs) were very popular features for a long time; but more recently, filter banks are becoming increasingly popular. In this post, I will discuss filter banks and MFCCs and why are filter banks becoming increasingly popular.

Computing filter banks and MFCCs involve somewhat the same procedure, where in both cases filter banks are computed and with a few more extra steps MFCCs can be obtained. In a nutshell, a signal goes through a pre-emphasis filter; then gets sliced into (overlapping) frames and a window function is applied to each frame; afterwards, we do a Fourier transform on each frame (or more specifically a Short-Time Fourier Transform) and calculate the power spectrum; and subsequently compute the filter banks. To obtain MFCCs, a Discrete Cosine Transform (DCT) is applied to the filter banks retaining a number of the resulting coefficients while the rest are discarded. A final step in both cases, is mean normalization.

## Setup

For this post, I used a 16-bit PCM wav file from [here](http://www.voiptroubleshooter.com/open_speech/american.html), called “OSR_us_000_0010_8k.wav”, which has a sampling frequency of 8000 Hz. The wav file is a clean speech signal comprising a single voice uttering some sentences with some pauses in-between. For simplicity, I used the first 3.5 seconds of the signal which corresponds roughly to the first sentence in the wav file.

I’ll be using Python 2.7.x, NumPy and SciPy. Some of the code used in this post is based on code available in this [repository](https://github.com/jameslyons/python_speech_features).

```python
import numpy
import scipy.io.wavfile
from scipy.fftpack import dct

sample_rate, signal = scipy.io.wavfile.read('OSR_us_000_0010_8k.wav')  # File assumed to be in the same directory
signal = signal[0:int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
```

The raw signal has the following form in the time domain:

![time_signal](./pictures source/time_signal.jpg)

## Pre-Emphasis

The first step is to apply a pre-emphasis filter on the signal to amplify the high frequencies. A pre-emphasis filter is useful in several ways: (1) balance the frequency spectrum since high frequencies usually have smaller magnitudes compared to lower frequencies, (2) avoid numerical problems during the Fourier transform operation and (3) may also improve the Signal-to-Noise Ratio (SNR).

The pre-emphasis filter can be applied to a signal xx using the first order filter in the following equation:



y(t)=x(t)−αx(t−1)y(t)=x(t)−αx(t−1)



which can be easily implemented using the following line, where typical values for the filter coefficient (αα) are 0.95 or 0.97, `pre_emphasis = 0.97`:

```
emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
```

[Pre-emphasis has a modest effect in modern systems](http://qr.ae/8GFgeI), mainly because most of the motivations for the pre-emphasis filter can be achieved using mean normalization (discussed later in this post) except for avoiding the Fourier transform numerical issues which should not be a problem in modern FFT implementations.

The signal after pre-emphasis has the following form in the time domain:![emphasized_time_signal](/home/jiang/桌面/About Python and some image algorithm/pictures source/emphasized_time_signal.jpg)



## Framing

After pre-emphasis, we need to split the signal into short-time frames. The rationale behind this step is that frequencies in a signal change over time, so in most cases it doesn’t make sense to do the Fourier transform across the entire signal in that we would lose the frequency contours of the signal over time. To avoid that, we can safely assume that frequencies in a signal are stationary over a very short period of time. Therefore, by doing a Fourier transform over this short-time frame, we can obtain a good approximation of the frequency contours of the signal by concatenating adjacent frames.

Typical frame sizes in speech processing range from 20 ms to 40 ms with 50% (+/-10%) overlap between consecutive frames. Popular settings are 25 ms for the frame size, `frame_size = 0.025` and a 10 ms stride (15 ms overlap), `frame_stride = 0.01`.

```python
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = numpy.zeros((pad_signal_length - signal_length))
pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(numpy.int32, copy=False)]
```

## Window

After slicing the signal into frames, we apply a window function such as the Hamming window to each frame. A Hamming window has the following form:



w[n]=0.54−0.46cos(2πnN−1)w[n]=0.54−0.46cos(2πnN−1)



where, 0≤n≤N−10≤n≤N−1, NN is the window length. Plotting the previous equation yields the following plot:![hamming_window](/home/jiang/桌面/About Python and some image algorithm/pictures source/hamming_window.jpg)

There are several reasons why we need to apply a window function to the frames, notably to counteract the assumption made by the FFT that the data is infinite and to reduce spectral leakage.

```
frames *= numpy.hamming(frame_length)
# frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
```

## Fourier-Transform and Power Spectrum

We can now do an NN-point FFT on each frame to calculate the frequency spectrum, which is also called Short-Time Fourier-Transform (STFT), where NN is typically 256 or 512, `NFFT = 512`; and then compute the power spectrum (periodogram) using the following equation:



P=|FFT(xi)|2NP=|FFT(xi)|2N



where, xixi is the ithith frame of signal xx. This could be implemented with the following lines:

```
mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
```

## Filter Banks

The final step to computing filter banks is applying triangular filters, typically 40 filters, `nfilt = 40` on a Mel-scale to the power spectrum to extract frequency bands. The Mel-scale aims to mimic the non-linear human ear perception of sound, by being more discriminative at lower frequencies and less discriminative at higher frequencies. We can convert between Hertz (ff) and Mel (mm) using the following equations:



m=2595log10(1+f700)m=2595log10⁡(1+f700)





f=700(10m/2595−1)f=700(10m/2595−1)



Each filter in the filter bank is triangular having a response of 1 at the center frequency and decrease linearly towards 0 till it reaches the center frequencies of the two adjacent filters where the response is 0, as shown in this figure:



# FIR filter

A Finite Impulse Response (FIR) filter is a type of filter used in digital signal processing. [Here are some key points about FIR filters](https://en.wikipedia.org/wiki/Finite_impulse_response)[1](https://en.wikipedia.org/wiki/Finite_impulse_response)[2](https://www.elprocus.com/fir-filter-for-digital-signal-processing/):

- [**Finite Impulse Response**: The term “finite impulse response” means that the filter’s impulse response (or response to any finite length input) is of finite duration because it settles to zero in finite time](https://en.wikipedia.org/wiki/Finite_impulse_response)[1](https://en.wikipedia.org/wiki/Finite_impulse_response).
- [**No Feedback Required**: FIR filters do not require feedback, which means that any rounding errors are not compounded by summed iterations](https://en.wikipedia.org/wiki/Finite_impulse_response)[1](https://en.wikipedia.org/wiki/Finite_impulse_response).
- [**Stability**: FIR filters are inherently stable, since the output is a sum of a finite number of finite multiples of the input values](https://en.wikipedia.org/wiki/Finite_impulse_response)[1](https://en.wikipedia.org/wiki/Finite_impulse_response).
- **Linear Phase**: FIR filters can easily be designed to be linear phase by making the coefficient sequence symmetric. [This property is sometimes desired for phase-sensitive applications](https://en.wikipedia.org/wiki/Finite_impulse_response)[1](https://en.wikipedia.org/wiki/Finite_impulse_response).
- [**Implementation**: The output of an FIR filter is a weighted sum of the most recent input values](https://en.wikipedia.org/wiki/Finite_impulse_response)[1](https://en.wikipedia.org/wiki/Finite_impulse_response). [This computation is also known as discrete convolution](https://en.wikipedia.org/wiki/Finite_impulse_response)[1](https://en.wikipedia.org/wiki/Finite_impulse_response).

[The coefficients used in these terms are commonly referred to as taps, based on the structure of a tapped delay line that in many implementations or block diagrams provides the delayed inputs to the multiplication operations](https://en.wikipedia.org/wiki/Finite_impulse_response)[1](https://en.wikipedia.org/wiki/Finite_impulse_response).

I hope this helps! Let me know if you have any other questions.

##  what is the reason we use FIR filter in digital processing, why we use this filter on the ECG.

[FIR filters are widely used in digital signal processing for several reasons](https://www.elprocus.com/fir-filter-for-digital-signal-processing/)[1](https://www.elprocus.com/fir-filter-for-digital-signal-processing/)[2](https://en.wikipedia.org/wiki/Finite_impulse_response)[3](https://adsantec.com/the-importance-of-filters-and-fir-filters-in-digital-signal-processing/)[4](https://dspguru.com/dsp/faqs/fir/basics/):

- [**Stability**: FIR filters are inherently stable, as the output is a sum of a finite number of finite multiples of the input values](https://www.elprocus.com/fir-filter-for-digital-signal-processing/)[1](https://www.elprocus.com/fir-filter-for-digital-signal-processing/)[2](https://en.wikipedia.org/wiki/Finite_impulse_response).
- [**No Feedback Required**: FIR filters do not require feedback, which means that any rounding errors are not compounded by summed iterations](https://www.elprocus.com/fir-filter-for-digital-signal-processing/)[1](https://www.elprocus.com/fir-filter-for-digital-signal-processing/)[2](https://en.wikipedia.org/wiki/Finite_impulse_response).
- **Linear Phase**: FIR filters can easily be designed to be linear phase by making the coefficient sequence symmetric. [This property is sometimes desired for phase-sensitive applications](https://www.elprocus.com/fir-filter-for-digital-signal-processing/)[1](https://www.elprocus.com/fir-filter-for-digital-signal-processing/)[2](https://en.wikipedia.org/wiki/Finite_impulse_response).
- [**Implementation Simplicity**: FIR filters are easy to implement and enable stable system designs](https://adsantec.com/the-importance-of-filters-and-fir-filters-in-digital-signal-processing/)[3](https://adsantec.com/the-importance-of-filters-and-fir-filters-in-digital-signal-processing/).
- [**Efficiency**: FIR filters can provide computational efficiency in certain applications](https://dspguru.com/dsp/faqs/fir/basics/)[4](https://dspguru.com/dsp/faqs/fir/basics/).

[As for ECG (Electrocardiogram) signal processing, FIR filters are particularly useful](https://link.springer.com/chapter/10.1007/978-981-19-1324-2_34)[5](https://link.springer.com/chapter/10.1007/978-981-19-1324-2_34)[6](https://www.gehealthcare.com/insights/article/a-guide-to-ecg-signal-filtering)[7](http://www.jscholaronline.org/articles/JBER/Signal-Processing.pdf):

- **Noise Removal**: ECG signals can be contaminated with various types of noise, such as power line interference, baseline drift, and movement artifacts. [FIR filters can effectively remove these noises](https://link.springer.com/chapter/10.1007/978-981-19-1324-2_34)[5](https://link.springer.com/chapter/10.1007/978-981-19-1324-2_34)[6](https://www.gehealthcare.com/insights/article/a-guide-to-ecg-signal-filtering).
- [**Preserving Signal Integrity**: Since the ECG signals are low-frequency signals (approximately 0.5–100 Hz), FIR filters can be designed to preserve the frequency components of interest while eliminating the unwanted noise](https://link.springer.com/chapter/10.1007/978-981-19-1324-2_34)[5](https://link.springer.com/chapter/10.1007/978-981-19-1324-2_34)[7](http://www.jscholaronline.org/articles/JBER/Signal-Processing.pdf).
- [**Baseline Wander Correction**: FIR high-pass filters can be used to estimate and remove the baseline wander in the ECG signal](http://www.jscholaronline.org/articles/JBER/Signal-Processing.pdf)[7](http://www.jscholaronline.org/articles/JBER/Signal-Processing.pdf).

I hope this helps! Let me know if you have any other questions.



​              

​              