import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa
from pydub import AudioSegment
import soundfile as sf



# def pass_filter(input_file, output_path, highcut=200, lowcut=5000, fs=44100, order=2):

    
#     # Read the input audio file
#     sample_rate, data = wav.read(input_path)

    
#     # # Design a high-pass filter
#     # nyquist = 0.5 * sample_rate
#     # normal_cutoff = highcut / nyquist
#     # b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
#     # filtered_data = signal.filtfilt(b, a, data)

#     nyquist = 0.5 * sample_rate
#     low = lowcut / nyquist
#     b, a = signal.butter(order, low, btype='low')
#     filtered_data = signal.filtfilt(b, a, data)

#     sf.write(output_path, filtered_data, sample_rate)


# def apply_reverb(input_file, output_path, impulse_response_file, dry_gain_reduction_db=0, wet_gain_db=00):
#     """
#     Apply reverb to an audio file with specified dry and wet gain adjustments.

#     Parameters:
#     - input_file: Path to the input audio file (WAV format).
#     - output_file: Path to save the processed audio file (WAV format).
#     - impulse_response_file: Path to the impulse response file (WAV format).
#     - dry_gain_reduction_db: Reduction of the dry gain in dB (default: 0 dB).
#     - wet_gain_db: Increase of the wet gain in dB (default: 0 dB).
#     """
    
#     # Read the input audio file
#     sample_rate, dry_signal = wav.read(input_path)
#     #sample_rate, dry_signal = wav.read(input_file)
    
#     # Read the impulse response file
#     ir_sample_rate, impulse_response = wav.read(impulse_response_file)
#     print(f"Input file sample rate: {sample_rate} Hz")
#     print(f"Impulse response file sample rate: {ir_sample_rate} Hz")

#     # Ensure the sample rates match
#     if sample_rate != ir_sample_rate:
#         print("Resampling impulse response to match input file sample rate...")
#         impulse_response = librosa.resample(impulse_response.astype(float), orig_sr=ir_sample_rate, target_sr=sample_rate)
    
#     # Normalize impulse response
#     impulse_response = impulse_response / np.max(np.abs(impulse_response))
    
#     # Apply convolution to get the wet signal
#     wet_signal = signal.convolve(dry_signal, impulse_response, mode='full')
    
#     # Trim or pad the wet signal to match the length of the dry signal
#     wet_signal = wet_signal[:len(dry_signal)]
    
#     # Adjust dry gain
#     dry_gain_factor = 10 ** (-dry_gain_reduction_db / 20)
#     dry_signal = dry_signal * dry_gain_factor
    
#     # Adjust wet gain
#     wet_gain_factor = 10 ** (wet_gain_db / 20)
#     wet_signal = wet_signal * wet_gain_factor
    
#     # Mix the dry and wet signals
#     mixed_signal = dry_signal + wet_signal
    
#     # Ensure the mixed signal is within the valid range
#     mixed_signal = np.clip(mixed_signal, -32768, 32767)

#     sf.write(output_path, mixed_signal, sample_rate)


def combine_audio_files(input_path, background_path, output_path, volume_db1=0, volume_db2=0):
    # Load audio files
    #fs1, audio1 = wav.read(input_path)
    audio2, fs2 = librosa.load(background_path, sr=None)
    fs1, audio1 = wav.read(input_path)

    
    # Normalize audio data to range [-1, 1]
    audio1 = audio1.astype(np.float32) / np.max(np.abs(audio1))
    audio2 = audio2.astype(np.float32) / np.max(np.abs(audio2))
    
    # Convert volume from dB to linear scale
    volume1 = 10**(volume_db1 / 20.0)
    volume2 = 10**(volume_db2 / 20.0)
    
    # Apply volume adjustment
    audio1 *= volume1
    audio2 *= volume2
    
    # Ensure both audios are of the same length
    min_length = min(len(audio1), len(audio2))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]

    print(f"input sampling rate: {fs1}")
    print(f"background sampling rate: {fs2}")
    # Ensure the sample rates match
    if fs2 != fs1:
        print("Resampling impulse response to match input file sample rate...")
        audio2 = librosa.resample(audio2.astype(float), orig_sr=fs2, target_sr=fs1)

    combined_audio = audio1 + audio2
    
    # Normalize combined audio to prevent clipping
    combined_audio = combined_audio / np.max(np.abs(combined_audio))
    
    # Scale back to integer PCM values
    combined_audio = np.int16(combined_audio * 32767)

    sf.write(output_path, combined_audio, fs1)

    print(f"Combined audio saved to {output_path}")

input_path ='generated_audio.wav'
background_path ='cuts_nv/chunk_2.mp3'
output_path = 'processed_audio.wav'
final_path = 'final_audio.wav'
impulse_response = "x06y06.wav"
# apply_reverb(input_path, output_path, impulse_response)
# pass_filter(output_path, output_path)
combine_audio_files(input_path, background_path, final_path, volume_db1 = 12, volume_db2 = 0)
