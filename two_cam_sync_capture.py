import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import threading
import time
import queue
from PIL import Image, ImageTk
import stapipy as st

# --- Configuration ---
DISPLAY_RESIZE_FACTOR = 0.3
TEMP_VIDEO_DIR = "temp_recordings"

# Ensure temp directory exists
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

class CameraWorker:
    """
    Handles the hardware interaction for a single camera datastream.
    """
    def __init__(self, device, datastream, index, st_converter):
        self.device = device
        self.datastream = datastream
        self.index = index
        self.st_converter = st_converter
        
        # Camera State
        self.is_running = False
        self.is_recording = False
        self.is_paused = False # If paused, we grab frames but don't record
        
        # Metrics
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0.0
        self.last_block_id = 0
        self.last_timestamp = 0
        self._prev_frame_time = 0
        
        # Video Writing
        self.video_writer = None
        self.temp_filename = os.path.join(TEMP_VIDEO_DIR, f"cam_{index}_temp.avi")
        
        # Image buffer for GUI
        self.latest_frame = None
        self.lock = threading.Lock()

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.datastream.start_acquisition()
            self.device.acquisition_start()
            self.start_time = time.time()
            self._prev_frame_time = time.time()

    def stop(self):
        self.is_running = False
        self.is_recording = False
        try:
            self.device.acquisition_stop()
            self.datastream.stop_acquisition()
        except:
            pass
        self.release_video()

    def start_recording(self):
        self.is_recording = True
        self.is_paused = False
        # Initialize Video Writer if not already
        if self.video_writer is None:
            # Note: We don't know frame size until first frame, 
            # so we might initialize writer in the grab loop
            pass

    def pause_recording(self):
        """Switches to Preview Mode (stops writing to file)"""
        self.is_paused = True

    def release_video(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def process_buffer(self, st_buffer):
        """Processes a raw StApi buffer"""
        if not st_buffer.info.is_image_present:
            return

        # 1. Update Metrics
        self.last_block_id = st_buffer.info.frame_id
        # Timestamp is usually in nanoseconds or tick counts depending on camera
        self.last_timestamp = st_buffer.info.timestamp 
        
        curr_time = time.time()
        self.fps = 1.0 / (curr_time - self._prev_frame_time) if (curr_time - self._prev_frame_time) > 0 else 0
        self._prev_frame_time = curr_time
        self.frame_count += 1

        # 2. Convert Image
        raw_image = st_buffer.get_image()
        converted_image = self.st_converter.convert(raw_image)
        data = converted_image.get_image_data()
        nparr = np.frombuffer(data, np.uint8)
        nparr = nparr.reshape(converted_image.height, converted_image.width, 3)

        # 3. Write to Video (If Recording and Not Paused)
        if self.is_recording and not self.is_paused:
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(
                    self.temp_filename, fourcc, 20.0, 
                    (converted_image.width, converted_image.height)
                )
            self.video_writer.write(nparr)

        # 4. Resize for GUI Display (Thread Safe Update)
        display_img = cv2.resize(nparr, None, fx=DISPLAY_RESIZE_FACTOR, fy=DISPLAY_RESIZE_FACTOR)
        # Convert BGR to RGB for Tkinter
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        with self.lock:
            self.latest_frame = display_img

class CameraSystem:
    """
    Manages the StApi System and CameraWorker threads.
    """
    def __init__(self):
        st.initialize()
        self.st_converter = st.create_converter(st.EStConverterType.PixelFormat)
        self.st_converter.destination_pixel_format = st.EStPixelFormatNamingConvention.BGR8
        self.st_system = st.create_system()
        self.workers = []
        self.global_running = False
        self.thread = None

    def discover_and_connect(self):
        """Connects to all available cameras."""
        st_devices = []
        while True:
            try:
                # Assuming create_first_device pops the device from available list
                st_devices.append(self.st_system.create_first_device())
            except:
                break
        
        print(f"Found {len(st_devices)} devices.")

        for i, device in enumerate(st_devices):
            datastream = device.create_datastream()
            worker = CameraWorker(device, datastream, i, self.st_converter)
            self.workers.append(worker)

    def start_grabbing_thread(self):
        """Start independent acquisition thread for each camera."""
        self.global_running = True
        for worker in self.workers:
            thread = threading.Thread(
                target=self._acquisition_loop_per_camera,
                args=(worker,),
                daemon=True
            )
            thread.start()

    def _acquisition_loop_per_camera(self, worker):
        """Each camera has its own acquisition loop."""
        worker.start()
        while self.global_running:
            if worker.datastream.is_grabbing:
                try:
                    with worker.datastream.retrieve_buffer(5000) as st_buffer:
                        worker.process_buffer(st_buffer)
                except:
                    pass

    def start_all_recording(self):
        """Start recording on all cameras."""
        for worker in self.workers:
            worker.start_recording()

    def pause_all_recording(self):
        """Pause recording on all cameras."""
        for worker in self.workers:
            worker.pause_recording()

    def save_all_videos(self):
        """Save videos from all cameras with a single directory selection."""
        was_recording = any(w.is_recording for w in self.workers)
        
        # Pause all recordings
        self.pause_all_recording()
        
        # Wait a moment for writers to release
        time.sleep(0.5)
        
        # Ask user for save directory once
        save_dir = filedialog.askdirectory(title="Select directory to save all videos")
        
        if not save_dir:
            # User cancelled, resume if was recording
            if was_recording:
                self.start_all_recording()
            return
        
        # Save all videos to the selected directory
        for worker in self.workers:
            self._save_single_video_to_dir(worker, save_dir)
        
        # Resume if it was recording
        if was_recording:
            self.start_all_recording()
            
    def _save_single_video_to_dir(self, worker, save_dir):
        """Save video from a single camera to specified directory."""
        src_path = worker.temp_filename
        if not os.path.exists(src_path):
            messagebox.showerror("Error", f"No video recorded yet for Camera {worker.index}.")
            return

        dest_path = os.path.join(save_dir, f"camera_{worker.index}_recording.avi")
        
        try:
            import shutil
            shutil.copy2(src_path, dest_path)
            
            # Reset temp writer
            worker.release_video()
            if os.path.exists(src_path):
                os.remove(src_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save Camera {worker.index}: {str(e)}")
            return
        
        print(f"Camera {worker.index} video saved to {dest_path}")

    def _save_single_video(self, worker):
        """Save video from a single camera."""
        was_recording = worker.is_recording
        worker.pause_recording()
        
        # Wait a moment for writer to release
        time.sleep(0.5) 
        
        src_path = worker.temp_filename
        if not os.path.exists(src_path):
            messagebox.showerror("Error", f"No video recorded yet for Camera {worker.index}.")
            return

        dest_path = filedialog.asksaveasfilename(
            defaultextension=".avi",
            filetypes=[("AVI files", "*.avi")],
            initialfile=f"camera_{worker.index}_recording.avi"
        )
        
        if dest_path:
            import shutil
            shutil.copy2(src_path, dest_path)
            messagebox.showinfo("Success", f"Camera {worker.index} video saved to {dest_path}")
            
            # Reset temp writer
            worker.release_video()
            if os.path.exists(src_path):
                os.remove(src_path)

        # Resume if it was recording
        if was_recording:
            worker.start_recording()
    
    def shutdown(self):
        self.global_running = False
        if self.thread:
            self.thread.join()
        for worker in self.workers:
            worker.stop()
        st.terminate()

# --- GUI Components ---

class CameraWidget(ttk.LabelFrame):
    """
    GUI Element for a SINGLE camera.
    """
    def __init__(self, parent, worker):
        super().__init__(parent, text=f"Camera {worker.index}: {worker.device.info.display_name}")
        self.worker = worker
        
        # Layout
        self.grid_columnconfigure(1, weight=1)
        
        # Image Display
        self.image_label = ttk.Label(self, text="Waiting for Stream...")
        self.image_label.grid(row=0, column=0, columnspan=3, padx=5, pady=5)
        
        # Stats Area
        self.stats_label = ttk.Label(self, text="FPS: 0 | Frames: 0 | Time: 00:00")
        self.stats_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=5)
        
        self.status_indicator = ttk.Label(self, text="Status: IDLE", foreground="gray")
        self.status_indicator.grid(row=2, column=0, columnspan=3, sticky="w", padx=5)

    def update_ui(self):
        # Update Image
        with self.worker.lock:
            if self.worker.latest_frame is not None:
                # Convert numpy array to PIL Image -> ImageTk
                img = Image.fromarray(self.worker.latest_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.image_label.imgtk = imgtk # Keep reference
                self.image_label.configure(image=imgtk)

        # Update Stats
        elapsed = int(time.time() - self.worker.start_time) if self.worker.is_running else 0
        mins, secs = divmod(elapsed, 60)
        
        stats_text = (f"FPS: {self.worker.fps:.1f} | "
                      f"Frames: {self.worker.frame_count} | "
                      f"Dur: {mins:02d}:{secs:02d} | "
                      f"ID: {self.worker.last_block_id}")
        self.stats_label.config(text=stats_text)

        # Update status
        if self.worker.is_recording and not self.worker.is_paused:
            self.status_indicator.config(text="Status: RECORDING", foreground="red")
        elif self.worker.is_paused:
            self.status_indicator.config(text="Status: PAUSED (Preview)", foreground="orange")
        else:
            self.status_indicator.config(text="Status: IDLE", foreground="gray")

class MultiCameraApp(tk.Tk):
    def __init__(self, cam_system):
        super().__init__()
        self.title("OperVu Multi-Camera Recorder")
        self.geometry("1000x900")
        self.cam_system = cam_system
        self.widgets = []

        # Header
        header = ttk.Label(self, text="StApi Camera Control System", font=("Arial", 16, "bold"))
        header.pack(pady=10)

        # Global Sync Status
        self.sync_label = ttk.Label(self, text="Sync Status: CHECKING...", font=("Arial", 12))
        self.sync_label.pack(pady=5)

        # Global Control Panel
        control_frame = ttk.LabelFrame(self, text="Global Controls")
        control_frame.pack(fill="x", padx=10, pady=10)

        self.btn_record = ttk.Button(control_frame, text="● Record All", command=self.on_record_all)
        self.btn_record.pack(side="left", padx=5, pady=5)
        
        self.btn_pause = ttk.Button(control_frame, text="⏸ Pause All", command=self.on_pause_all)
        self.btn_pause.pack(side="left", padx=5, pady=5)

        self.btn_save = ttk.Button(control_frame, text="💾 Save All Videos", command=self.on_save_all)
        self.btn_save.pack(side="left", padx=5, pady=5)

        # Camera Container
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True, padx=10, pady=10)

        self.create_widgets()
        
        # Start Update Loop
        self.after(30, self.update_loop)

        # Handle Close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # Dynamic Grid based on camera count
        for i, worker in enumerate(self.cam_system.workers):
            widget = CameraWidget(self.container, worker)
            widget.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="nsew")
            self.widgets.append(widget)

    def on_record_all(self):
        """Start recording on all cameras."""
        self.cam_system.start_all_recording()
        self.btn_record.config(state="disabled")
        self.btn_pause.config(state="normal")

    def on_pause_all(self):
        """Pause recording on all cameras."""
        self.cam_system.pause_all_recording()
        self.btn_record.config(state="normal")
        self.btn_pause.config(state="disabled")

    def on_save_all(self):
        """Save videos from all cameras."""
        was_recording = any(w.is_recording for w in self.cam_system.workers)
        self.cam_system.save_all_videos()
        if was_recording:
            self.on_record_all()

    def update_loop(self):
        # 1. Update individual cameras
        timestamps = []
        for widget in self.widgets:
            widget.update_ui()
            timestamps.append(widget.worker.last_timestamp)

        # 2. Check Synchronization
        if len(timestamps) > 1 and all(t > 0 for t in timestamps):
            # Calculate max difference between timestamps
            # Note: Timestamp units depend on camera (ns vs ticks). Assuming ns or close units.
            # If using blocks/FrameIDs for sync:
            block_ids = [w.worker.last_block_id for w in self.widgets]
            
            # Simple check: Are frame IDs identical?
            # Or check time diff
            diff = max(timestamps) - min(timestamps)
            
            # Heuristic: If diff is small (arbitrary unit check needed based on hardware)
            # Just displaying the raw diff for now
            status = "SYNCED" if diff < 1000000 else "UNSYNCED" # Example threshold
            
            self.sync_label.config(
                text=f"Sync Status: {status} (Diff: {diff} | FrameIDs: {block_ids})",
                foreground="green" if status == "SYNCED" else "red"
            )

        self.after(30, self.update_loop)

    def on_close(self):
        if messagebox.askokcancel("Quit", "Do you want to stop acquisition and quit?"):
            self.cam_system.shutdown()
            self.destroy()

# --- Main Execution ---

if __name__ == "__main__":
    try:
        # 1. Initialize Hardware
        system = CameraSystem()
        print("Connecting to cameras...")
        system.discover_and_connect()
        
        if not system.workers:
            print("No cameras found! Exiting.")
            exit()

        # 2. Start Grabbing Thread
        system.start_grabbing_thread()

        # 3. Start GUI
        app = MultiCameraApp(system)
        app.mainloop()

    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()