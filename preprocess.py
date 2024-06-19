from pydub import AudioSegment
import os


def cut_mp3_into_chunks(mp3_path, chunk_length_ms, output_dir):
    # Load the mp3 file
    audio = AudioSegment.from_mp3(mp3_path)
    
    # Calculate the number of chunks
    total_length_ms = len(audio)
    num_chunks = total_length_ms // chunk_length_ms
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Split the audio into chunks and save them
    for i in range(num_chunks):
        start_time = i * chunk_length_ms
        end_time = start_time + chunk_length_ms
        chunk = audio[start_time:end_time]
        chunk_filename = os.path.join(output_dir, f'chunk_{i+1}.mp3')
        chunk.export(chunk_filename, format="mp3")
    
    # If there's a remaining part that doesn't fit into the chunk_length
    if total_length_ms % chunk_length_ms != 0:
        start_time = num_chunks * chunk_length_ms
        chunk = audio[start_time:]
        chunk_filename = os.path.join(output_dir, f'chunk_{num_chunks + 1}.mp3')
        chunk.export(chunk_filename, format="mp3")

# Example usage
mp3_path1 = 'vocals.mp3'  # replace with your file path
mp3_path2 = 'no_vocals.mp3'  # replace with your file path
chunk_length_ms = 30000  # 30 seconds


output_dir1 = 'cuts_v'  # directory to save the chunks
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)
output_dir2 = 'cuts_nv'  # directory to save the chunks
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)
cut_mp3_into_chunks(mp3_path1, chunk_length_ms, output_dir1)
cut_mp3_into_chunks(mp3_path2, chunk_length_ms, output_dir2)