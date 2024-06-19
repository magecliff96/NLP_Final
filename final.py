import librosa
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import soundfile as sf
import cv2
from librosa.display import specshow

# Load audio and convert to spectrogram
def audio_to_spectrogram(y, sr):
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db, sr

# Apply Gaussian filter to the spectrogram
def apply_horizontal_gaussian_blur(S_db, ksize=15, sigma=100):
    return cv2.GaussianBlur(S_db, (ksize, 1), sigma)

def filter_spectrogram(S_db):
    median_value = np.median(S_db)
    threshold = 0.7 * median_value
    S_db_filtered = np.where(S_db < threshold, np.min(S_db), S_db)
    return S_db_filtered

# Convert spectrogram back to audio
def spectrogram_to_audio(S_db, sr):
    S = librosa.db_to_amplitude(S_db)
    y = librosa.griffinlim(S)
    return y

def save_spectrogram(S_db, sr, img_path, npy_path):
    plt.figure(figsize=(10, 4))
    specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(img_path)
    plt.close()
    np.save(npy_path, S_db)

def normalize_audio(y):
    return y / np.max(np.abs(y))

def apply_gain(y, gain_db):
    factor = 10**(gain_db / 20)
    return y * factor


# Example spectrogram (using a sample audio file)
audio_path = 'cuts_v/chunk_2.mp3'  # replace with actual path
y, sr = librosa.load(audio_path, sr=None)
# pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
# plt.figure(figsize=(10, 6))
# plt.imshow(pitches[:100, :], aspect='auto', interpolation='nearest', origin='lower', cmap='viridis')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Pitch Spectrogram')
# plt.xlabel('Time Frames')
# plt.ylabel('Frequency Bins')
# plt.show()

spectrogram, sr = audio_to_spectrogram(y, sr)
spectrogram_filtered = apply_horizontal_gaussian_blur(filter_spectrogram(spectrogram))


texture_path = 'cuts_nv/chunk_2.mp3'
y_t, sr_t = librosa.load(texture_path, sr=None)
t_spectrogram, t_sr = audio_to_spectrogram(y_t, sr_t)
# t_spectrogram_blurred = apply_horizontal_gaussian_blur(filter_spectrogram(t_spectrogram))
# nv_texture = t_spectrogram - t_spectrogram_blurred


# Define the CNN model with Dilated Convolutions
class MusicDIP(nn.Module):
    def __init__(self):
        super(MusicDIP, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 9), padding=(0, 4), dilation=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 9), padding=(0, 8), dilation=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 9), padding=(0, 16), dilation=4)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(1, 9), padding=(0, 8), dilation=2)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=3, padding=1, dilation=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x

# Initialize the model and move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MusicDIP().to(device)

# Prepare the blurred spectrogram for input and target
input_spectrogram = torch.tensor(spectrogram_filtered).unsqueeze(0).unsqueeze(0).to(device)
target_spectrogram = torch.tensor(spectrogram).unsqueeze(0).unsqueeze(0).to(device)
#target_texture = torch.tensor(nv_texture).unsqueeze(0).unsqueeze(0).to(device)

# Training function
def train(model, input_spectrogram, target_spectrogram, num_iterations=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=1e-1)
    criterion = nn.MSELoss()
    loss = 1e3;i=0
    
    while loss>90:
        i+=1
        optimizer.zero_grad()
        output = model(input_spectrogram)
        # output_np = output.cpu().detach().numpy().squeeze()
        # output_blurred = apply_horizontal_gaussian_blur(output_np)
        # texture = output_np - output_blurred
        # texture = torch.tensor(texture).unsqueeze(0).unsqueeze(0).to(device)
        loss = criterion(output, target_spectrogram)# + criterion(texture, target_texture)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'Iteration {i}, Loss: {loss.item()}')
        if i > num_iterations:
            break

# Train the model
train(model, input_spectrogram, target_spectrogram)

# Generate output spectrogram
with torch.no_grad():
    output_spectrogram = model(input_spectrogram).cpu().detach().numpy().squeeze()
save_spectrogram(output_spectrogram, sr, 'generated_spectrogram.png', 'generated_spectrogram.npy')

# Convert generated spectrogram back to audio
output_audio = spectrogram_to_audio(output_spectrogram, sr)
output_audio_normalized = normalize_audio(output_audio)
output_audio_normalized = apply_gain(output_audio_normalized, 0)  # Apply 10 dB gain
# Save the generated audio
sf.write('generated_audio.wav', output_audio_normalized, sr)


# Display the original and generated spectrograms
plt.figure(figsize=(16, 8))
plt.subplot(2, 2, 1)
specshow(spectrogram, sr=sr, x_axis='time', y_axis='log')
plt.title('Original Spectrogram')
plt.subplot(2, 2, 2)
specshow(spectrogram_filtered, sr=sr, x_axis='time', y_axis='log')
plt.title('Input Spectrogram')
plt.subplot(2, 2, 3)
specshow(output_spectrogram, sr=sr, x_axis='time', y_axis='log') #output
plt.title('Generated Spectrogram')
# plt.subplot(2, 3, 4)
# specshow(t_spectrogram, sr=sr, x_axis='time', y_axis='log')
# plt.title('Nv Spectrogram')
# plt.subplot(2, 3, 5)
# specshow(nv_texture, sr=sr, x_axis='time', y_axis='log')
# plt.title('Nv Texture')
plt.show()


# audio_segment = AudioSegment(
# output_audio_normalized.tobytes(), 
# frame_rate=sample_rate,
# sample_width=2, 
# channels=1)
# # Write combined audio to file
# audio_segment.export('generated_audio.mp3', format="mp3")
### Post processing ###