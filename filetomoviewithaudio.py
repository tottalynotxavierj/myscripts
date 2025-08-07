#!/usr/bin/env python3

import os
import numpy as np
import cv2
from moviepy.editor import ImageSequenceClip, AudioClip

def file_to_frames(file_path, frame_size=256):
    """Read file and convert it into fixed-size color frames (RGB)."""
    chunk_size = frame_size * frame_size * 3  # RGB 3 channels

    frames = []
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            # Pad chunk to full frame size
            if len(chunk) < chunk_size:
                chunk += b'\x00' * (chunk_size - len(chunk))

            img_array = np.frombuffer(chunk, dtype=np.uint8).reshape((frame_size, frame_size, 3))
            # Convert from RGB to BGR for OpenCV compatibility
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            frames.append(img_bgr)
    return frames

def generate_audio_from_file(file_path, duration, sample_rate=44100):
    """
    Generate an audio waveform from the data in the file.
    """
    with open(file_path, 'rb') as f:
        data = f.read()

    data_array = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    data_normalized = (data_array - 128) / 128.0

    total_samples = int(sample_rate * duration)
    repeats = int(np.ceil(total_samples / len(data_normalized)))
    waveform = np.tile(data_normalized, repeats)[:total_samples]

    def make_frame(t):
        t = np.atleast_1d(t)
        indices = (t * sample_rate).astype(int)
        indices = np.clip(indices, 0, total_samples - 1)
        samples = waveform[indices]
        return samples.astype(np.float32)

    return AudioClip(make_frame, duration=duration, fps=sample_rate)

def main():
    file_path = input("Enter the path to the file you want to convert to video: ").strip()
    if not os.path.isfile(file_path):
        print("File does not exist.")
        return

    print("Creating frames from the file...")
    frames = file_to_frames(file_path, frame_size=256)
    print(f"Generated {len(frames)} frames.")

    output_video = input("Enter the output video filename (e.g., output.mp4): ").strip()
    fps = int(input("Enter frames per second (e.g., 10): ").strip())

    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    clip = ImageSequenceClip(rgb_frames, fps=fps)

    total_duration = len(frames) / fps
    print(f"Generating audio from file data for {total_duration:.2f} seconds...")

    audio_clip = generate_audio_from_file(file_path, duration=total_duration, sample_rate=44100)

    # Set the generated audio
    clip = clip.set_audio(audio_clip)

    # Save the final video
    clip.write_videofile(output_video, codec='libx264', audio_codec='aac')
    print(f"Video with generated audio saved to {output_video}")

if __name__ == "__main__":
    main()
