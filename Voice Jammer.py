import sounddevice as sd
import numpy as np
import time
import queue
import threading
import librosa
import tkinter as tk
from tkinter import ttk, messagebox
from scipy import signal

def select_devices():
    """
    Opens a window to select input and output devices.
    Returns tuple of (input_device_id, output_device_id)
    """
    devices = sd.query_devices()
    
    # First select input device
    input_window = tk.Toplevel()
    input_window.title("Select Input Device")
    input_window.geometry("500x300")
    
    # Create input frame
    input_frame = ttk.LabelFrame(input_window, text="Select Input Device (Discord Audio)", padding="5")
    input_frame.pack(fill="x", padx=5, pady=5)
    
    selected_input = [None]
    
    # Create input list
    input_list = tk.Listbox(input_frame, height=10, selectmode=tk.BROWSE)
    input_list.pack(fill="x", padx=5, pady=5)
    
    # Populate input list
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_list.insert(tk.END, f"{device['name']}")
            input_devices.append(i)
    
    def on_input_select():
        try:
            input_idx = input_list.curselection()[0]
            selected_input[0] = input_devices[input_idx]
            input_window.destroy()
        except IndexError:
            messagebox.showerror("Error", "Please select an input device")
    
    ttk.Button(input_window, text="Select Input", command=on_input_select).pack(pady=10)
    
    # Wait for input window to close
    input_window.transient(input_window.master)
    input_window.grab_set()
    input_window.master.wait_window(input_window)
    
    if selected_input[0] is None:
        return None, None
    
    # Then select output device
    output_window = tk.Toplevel()
    output_window.title("Select Output Device")
    output_window.geometry("500x300")
    
    # Create output frame
    output_frame = ttk.LabelFrame(output_window, text="Select Output Device (To Voicemeeter)", padding="5")
    output_frame.pack(fill="x", padx=5, pady=5)
    
    selected_output = [None]
    
    # Create output list
    output_list = tk.Listbox(output_frame, height=10, selectmode=tk.BROWSE)
    output_list.pack(fill="x", padx=5, pady=5)
    
    # Populate output list
    output_devices = []
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            output_list.insert(tk.END, f"{device['name']}")
            output_devices.append(i)
    
    def on_output_select():
        try:
            output_idx = output_list.curselection()[0]
            selected_output[0] = output_devices[output_idx]
            output_window.destroy()
        except IndexError:
            messagebox.showerror("Error", "Please select an output device")
    
    ttk.Button(output_window, text="Select Output", command=on_output_select).pack(pady=10)
    
    # Wait for output window to close
    output_window.transient(output_window.master)
    output_window.grab_set()
    output_window.master.wait_window(output_window)
    
    return selected_input[0], selected_output[0]

class AudioEffects:
    def __init__(self):
        self.pitch_shift = 0
        self.echo_delay = 0.0
        self.distortion = 0.0
        self.volume_boost = 2.0
        self.delay_ms = 500
        
    def apply_pitch_shift(self, audio, sample_rate):
        if self.pitch_shift == 0:
            return audio
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # Use librosa for pitch shifting
        return librosa.effects.pitch_shift(
            y=audio.astype(np.float32),
            sr=sample_rate,
            n_steps=self.pitch_shift
        ).reshape(-1, 1)
    
    def apply_echo(self, audio):
        if self.echo_delay == 0:
            return audio
            
        # Create delayed version of the signal
        delay_samples = int(44100 * self.echo_delay)  # assuming 44.1kHz sample rate
        if delay_samples == 0:
            return audio
            
        delayed = np.zeros_like(audio)
        delayed[delay_samples:] = audio[:-delay_samples] * 0.6  # 0.6 is echo volume
        
        # Mix original and delayed signals
        return audio + delayed
    
    def apply_distortion(self, audio):
        if self.distortion == 0:
            return audio
            
        # Simple waveshaping distortion
        threshold = 1.0 - self.distortion
        return np.clip(audio * (1 + self.distortion), -threshold, threshold)

class VoiceJammerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Jammer Control Panel")
        self.root.geometry("400x600")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Device selection frame
        device_frame = ttk.LabelFrame(main_frame, text="Audio Devices", padding="5")
        device_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(device_frame, text="Select Devices", command=self.select_devices).grid(row=0, column=0, pady=5)
        self.device_label = ttk.Label(device_frame, text="No devices selected")
        self.device_label.grid(row=1, column=0, pady=5)
        
        # Effects frame
        effects_frame = ttk.LabelFrame(main_frame, text="Effects Controls", padding="5")
        effects_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Pitch control
        ttk.Label(effects_frame, text="Pitch Shift").grid(row=0, column=0, pady=2)
        self.pitch_var = tk.DoubleVar(value=0)
        pitch_slider = ttk.Scale(effects_frame, from_=-12, to=12, variable=self.pitch_var,
                               orient=tk.HORIZONTAL, command=self.update_pitch)
        pitch_slider.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Delay control
        ttk.Label(effects_frame, text="Delay (ms)").grid(row=2, column=0, pady=2)
        self.delay_var = tk.DoubleVar(value=500)
        delay_slider = ttk.Scale(effects_frame, from_=0, to=1000, variable=self.delay_var,
                               orient=tk.HORIZONTAL, command=self.update_delay)
        delay_slider.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Echo control
        ttk.Label(effects_frame, text="Echo").grid(row=4, column=0, pady=2)
        self.echo_var = tk.DoubleVar(value=0)
        echo_slider = ttk.Scale(effects_frame, from_=0, to=1, variable=self.echo_var,
                              orient=tk.HORIZONTAL, command=self.update_echo)
        echo_slider.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Distortion control
        ttk.Label(effects_frame, text="Distortion").grid(row=6, column=0, pady=2)
        self.distortion_var = tk.DoubleVar(value=0)
        distortion_slider = ttk.Scale(effects_frame, from_=0, to=1, variable=self.distortion_var,
                                    orient=tk.HORIZONTAL, command=self.update_distortion)
        distortion_slider.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Volume control
        ttk.Label(effects_frame, text="Volume").grid(row=8, column=0, pady=2)
        self.volume_var = tk.DoubleVar(value=2.0)
        volume_slider = ttk.Scale(effects_frame, from_=0, to=10, variable=self.volume_var,
                                orient=tk.HORIZONTAL, command=self.update_volume)
        volume_slider.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=10)
        
        ttk.Button(button_frame, text="Reset Effects", command=self.reset_effects).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Clear Queue", command=self.clear_queue).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Start", command=self.start_audio).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Stop", command=self.stop_audio).grid(row=0, column=3, padx=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.grid(row=0, column=0)
        
        # Initialize audio components
        self.effects = AudioEffects()
        self.audio_queue = queue.Queue()
        self.input_device = None
        self.output_device = None
        self.stream = None
        self.is_running = False

    def update_pitch(self, value):
        self.effects.pitch_shift = float(value)
        
    def update_delay(self, value):
        self.effects.delay_ms = float(value)
        self.clear_queue()
        
    def update_echo(self, value):
        self.effects.echo_delay = float(value)
        
    def update_distortion(self, value):
        self.effects.distortion = float(value)
        
    def update_volume(self, value):
        self.effects.volume_boost = float(value)
        
    def reset_effects(self):
        self.pitch_var.set(0)
        self.delay_var.set(500)
        self.echo_var.set(0)
        self.distortion_var.set(0)
        self.volume_var.set(2.0)
        self.effects = AudioEffects()
        self.clear_queue()
        
    def clear_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.status_label.config(text="Queue cleared")
        
    def select_devices(self):
        self.input_device, self.output_device = select_devices()
        if self.input_device is not None and self.output_device is not None:
            input_name = sd.query_devices(self.input_device)['name']
            output_name = sd.query_devices(self.output_device)['name']
            self.device_label.config(text=f"In: {input_name}\nOut: {output_name}")
            
    def audio_callback(self, indata, frames, time, status):
        if status:
            self.status_label.config(text=f"Input status: {status}")
        try:
            self.audio_queue.put(indata.copy())
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")
            
    def start_audio(self):
        if self.input_device is None or self.output_device is None:
            self.status_label.config(text="Please select devices first")
            return
            
        if not self.is_running:
            try:
                self.stream = sd.InputStream(
                    device=self.input_device,
                    callback=self.audio_callback,
                    channels=1,
                    samplerate=44100,
                    blocksize=2048
                )
                self.stream.start()
                self.is_running = True
                
                # Start playback thread
                self.playback_thread = threading.Thread(target=self.playback, daemon=True)
                self.playback_thread.start()
                
                self.status_label.config(text="Audio processing started")
            except Exception as e:
                self.status_label.config(text=f"Error starting: {e}")
                
    def stop_audio(self):
        if self.is_running:
            self.is_running = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
            self.clear_queue()
            self.status_label.config(text="Audio processing stopped")
            
    def playback(self):
        delay_samples = int(self.effects.delay_ms * (44100 / 1000))
        buffer = np.zeros((delay_samples, 1), dtype=np.float32)
        
        with sd.OutputStream(device=self.output_device, channels=1,
                           samplerate=44100, blocksize=2048) as output_stream:
            while self.is_running:
                try:
                    data = self.audio_queue.get(timeout=0.1)
                    
                    current_delay_samples = int(self.effects.delay_ms * (44100 / 1000))
                    if current_delay_samples != len(buffer):
                        new_buffer = np.zeros((current_delay_samples, 1), dtype=np.float32)
                        new_buffer[:min(len(buffer), len(new_buffer))] = buffer[:min(len(buffer), len(new_buffer))]
                        buffer = new_buffer
                        
                    buffer = np.concatenate((buffer, data), axis=0)
                    outdata = buffer[:len(data)]
                    buffer = buffer[len(data):]
                    
                    # Apply effects chain
                    outdata = self.effects.apply_pitch_shift(outdata, 44100)
                    outdata = self.effects.apply_echo(outdata)
                    outdata = self.effects.apply_distortion(outdata)
                    outdata = np.clip(outdata * self.effects.volume_boost, -1, 1)
                    
                    output_stream.write(outdata)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.status_label.config(text=f"Playback error: {e}")

def main():
    root = tk.Tk()
    app = VoiceJammerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()