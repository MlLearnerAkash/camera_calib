import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os

class HomographyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linux Homography Calibration Tool")
        self.root.geometry("1200x800")
        
        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # State Variables
        self.mode = None # 'online' or 'offline'
        self.board_size = (9, 6) # Internal corners (columns, rows)
        self.captures = [] # List of tuples (img1, img2)
        self.obj_points_list = [] # 3D points in real world space
        self.img_points1_list = [] # 2D points in image plane 1
        self.img_points2_list = [] # 2D points in image plane 2
        
        self.cam1_idx = 0
        self.cam2_idx = 2 # Usually 0 is laptop cam, 2 is external on Linux often, or 0 and 1
        self.cap1 = None
        self.cap2 = None
        self.is_streaming = False
        
        self.homography_matrix = None
        
        # --- GUI Layout ---
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.setup_start_screen()

    def setup_start_screen(self):
        self.clear_frame(self.main_container)
        
        lbl = ttk.Label(self.main_container, text="Homography Calibration Tool", font=("Helvetica", 24, "bold"))
        lbl.pack(pady=40)
        
        btn_frame = ttk.Frame(self.main_container)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Online Mode (Live Cameras)", command=lambda: self.config_screen('online')).pack(side=tk.LEFT, padx=20, ipadx=20, ipady=10)
        ttk.Button(btn_frame, text="Offline Mode (Upload Images)", command=lambda: self.config_screen('offline')).pack(side=tk.LEFT, padx=20, ipadx=20, ipady=10)

    def config_screen(self, mode):
        self.mode = mode
        self.clear_frame(self.main_container)
        
        ttk.Label(self.main_container, text="Configuration", font=("Helvetica", 18)).pack(pady=20)
        
        form_frame = ttk.Frame(self.main_container)
        form_frame.pack(pady=10)
        
        # Checkerboard Config
        ttk.Label(form_frame, text="Checkerboard Pattern Size (Internal Corners):").grid(row=0, column=0, sticky="e", padx=5)
        
        dim_frame = ttk.Frame(form_frame)
        dim_frame.grid(row=0, column=1, sticky="w")
        
        self.cols_var = tk.IntVar(value=9)
        self.rows_var = tk.IntVar(value=6)
        
        ttk.Entry(dim_frame, textvariable=self.cols_var, width=5).pack(side=tk.LEFT)
        ttk.Label(dim_frame, text="x").pack(side=tk.LEFT)
        ttk.Entry(dim_frame, textvariable=self.rows_var, width=5).pack(side=tk.LEFT)
        
        # Camera Config (Only for Online)
        if mode == 'online':
            ttk.Label(form_frame, text="Camera Indices (1 & 2):").grid(row=1, column=0, sticky="e", padx=5, pady=10)
            cam_frame = ttk.Frame(form_frame)
            cam_frame.grid(row=1, column=1, sticky="w", pady=10)
            
            self.cam1_entry = tk.Entry(cam_frame, width=5)
            self.cam1_entry.insert(0, "0")
            self.cam1_entry.pack(side=tk.LEFT, padx=2)
            
            self.cam2_entry = tk.Entry(cam_frame, width=5)
            self.cam2_entry.insert(0, "1") # Default to 1, user might need to change to 2 or higher
            self.cam2_entry.pack(side=tk.LEFT, padx=2)

        ttk.Button(self.main_container, text="Start Session", command=self.init_session).pack(pady=30)

    def init_session(self):
        try:
            c = self.cols_var.get()
            r = self.rows_var.get()
            self.board_size = (c, r)
        except:
            messagebox.showerror("Error", "Invalid dimensions")
            return

        if self.mode == 'online':
            try:
                self.cam1_idx = int(self.cam1_entry.get())
                self.cam2_idx = int(self.cam2_entry.get())
                self.setup_online_interface()
            except ValueError:
                messagebox.showerror("Error", "Camera indices must be integers")
        else:
            self.setup_offline_interface()

    # ---------------- ONLINE MODE ----------------
    def setup_online_interface(self):
        self.clear_frame(self.main_container)
        
        # Video Frame
        video_frame = ttk.Frame(self.main_container)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.lbl_cam1 = ttk.Label(video_frame, text="Cam 1 Loading...")
        self.lbl_cam1.grid(row=0, column=0, padx=5, pady=5)
        
        self.lbl_cam2 = ttk.Label(video_frame, text="Cam 2 Loading...")
        self.lbl_cam2.grid(row=0, column=1, padx=5, pady=5)
        
        # Control Frame
        ctrl_frame = ttk.Frame(self.main_container)
        ctrl_frame.pack(fill=tk.X, pady=10)
        
        self.status_lbl = ttk.Label(ctrl_frame, text="Pairs Captured: 0/4 (Minimum)", foreground="red")
        self.status_lbl.pack(side=tk.LEFT, padx=20)
        
        self.btn_capture = ttk.Button(ctrl_frame, text="Capture Pair", command=self.capture_frame)
        self.btn_capture.pack(side=tk.LEFT, padx=10)
        
        self.btn_calc = ttk.Button(ctrl_frame, text="Calculate Homography", state=tk.DISABLED, command=self.calculate_homography)
        self.btn_calc.pack(side=tk.RIGHT, padx=20)

        # Start Streams
        self.cap1 = cv2.VideoCapture(self.cam1_idx)
        self.cap2 = cv2.VideoCapture(self.cam2_idx)
        
        if not self.cap1.isOpened() or not self.cap2.isOpened():
            messagebox.showerror("Error", "Could not open one or both cameras.")
            return

        self.is_streaming = True
        self.update_streams()

    def update_streams(self):
        if not self.is_streaming:
            return
            
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        
        if ret1 and ret2:
            self.current_frame1 = frame1
            self.current_frame2 = frame2
            
            # Resize for display
            disp1 = self.convert_cv_to_tk(frame1)
            disp2 = self.convert_cv_to_tk(frame2)
            
            self.lbl_cam1.configure(image=disp1)
            self.lbl_cam1.image = disp1
            self.lbl_cam2.configure(image=disp2)
            self.lbl_cam2.image = disp2
        
        self.root.after(20, self.update_streams)

    def capture_frame(self):
        if hasattr(self, 'current_frame1') and hasattr(self, 'current_frame2'):
            # Flash effect or log
            img1 = self.current_frame1.copy()
            img2 = self.current_frame2.copy()
            
            # Quick check if board is visible before adding
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            ret1, _ = cv2.findChessboardCorners(gray1, self.board_size, None)
            ret2, _ = cv2.findChessboardCorners(gray2, self.board_size, None)
            
            if ret1 and ret2:
                self.captures.append((img1, img2))
                count = len(self.captures)
                self.status_lbl.config(text=f"Pairs Captured: {count}", foreground="green" if count >= 4 else "orange")
                messagebox.showinfo("Success", "Checkerboard detected and captured!")
                
                if count >= 4:
                    self.btn_calc.config(state=tk.NORMAL)
            else:
                messagebox.showwarning("Warning", "Checkerboard NOT detected in both frames. Try moving the board.")

    # ---------------- OFFLINE MODE ----------------
    def setup_offline_interface(self):
        self.clear_frame(self.main_container)
        
        ttk.Label(self.main_container, text="Offline Mode: Upload Image Pairs", font=("Helvetica", 14)).pack(pady=20)
        
        btn_frame = ttk.Frame(self.main_container)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Select Cam 1 Images", command=self.load_cam1_images).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Select Cam 2 Images", command=self.load_cam2_images).pack(side=tk.LEFT, padx=10)
        
        self.file_list_lbl = ttk.Label(self.main_container, text="No images loaded.")
        self.file_list_lbl.pack(pady=20)
        
        self.btn_calc_offline = ttk.Button(self.main_container, text="Process & Calculate", state=tk.DISABLED, command=self.process_offline_images)
        self.btn_calc_offline.pack(pady=10)
        
        self.files1 = []
        self.files2 = []

    def load_cam1_images(self):
        files = filedialog.askopenfilenames(title="Select Camera 1 Images", filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if files:
            self.files1 = sorted(list(files))
            self.check_files()

    def load_cam2_images(self):
        files = filedialog.askopenfilenames(title="Select Camera 2 Images", filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if files:
            self.files2 = sorted(list(files))
            self.check_files()

    def check_files(self):
        if len(self.files1) > 0 and len(self.files2) > 0:
            if len(self.files1) != len(self.files2):
                self.file_list_lbl.config(text=f"Mismatch! Cam1: {len(self.files1)}, Cam2: {len(self.files2)}")
                self.btn_calc_offline.config(state=tk.DISABLED)
            else:
                self.file_list_lbl.config(text=f"Ready: {len(self.files1)} pairs loaded.")
                if len(self.files1) >= 4:
                    self.btn_calc_offline.config(state=tk.NORMAL)
                else:
                    self.file_list_lbl.config(text=f"Need at least 4 pairs. Loaded: {len(self.files1)}")

    def process_offline_images(self):
        # Load images into memory
        self.captures = []
        for f1, f2 in zip(self.files1, self.files2):
            img1 = cv2.imread(f1)
            img2 = cv2.imread(f2)
            if img1 is not None and img2 is not None:
                self.captures.append((img1, img2))
        
        self.calculate_homography()

    # ---------------- CALCULATION LOGIC ----------------
    def calculate_homography(self):
        if self.is_streaming:
            self.is_streaming = False
            self.cap1.release()
            self.cap2.release()

        self.img_points1_list = []
        self.img_points2_list = []
        self.good_captures = []
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        loading_popup = tk.Toplevel(self.root)
        ttk.Label(loading_popup, text="Processing Images... Please Wait").pack(padx=20, pady=20)
        self.root.update()

        for (img1, img2) in self.captures:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            ret1, corners1 = cv2.findChessboardCorners(gray1, self.board_size, None)
            ret2, corners2 = cv2.findChessboardCorners(gray2, self.board_size, None)
            
            if ret1 and ret2:
                corners1_sub = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                corners2_sub = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
                
                self.img_points1_list.append(corners1_sub)
                self.img_points2_list.append(corners2_sub)
                self.good_captures.append((img1, img2, corners1_sub, corners2_sub))
        
        loading_popup.destroy()

        if len(self.img_points1_list) < 4:
            messagebox.showerror("Error", "Not enough valid checkerboard pairs found (Need 4+).")
            # If online, restart stream
            if self.mode == 'online':
                self.setup_online_interface()
            return

        # Stack points for Homography
        src_pts = np.concatenate(self.img_points1_list).reshape(-1, 2)
        dst_pts = np.concatenate(self.img_points2_list).reshape(-1, 2)
        
        # Calculate Homography (Cam1 -> Cam2)
        self.H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        self.show_results_dashboard(src_pts, dst_pts)

    # ---------------- RESULTS / STATS FOR NERDS ----------------
    def show_results_dashboard(self, src_pts, dst_pts):
        self.clear_frame(self.main_container)
        
        # Calculate Reprojection Error
        projected_pts_2 = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), self.H).reshape(-1, 2)
        error = np.sqrt(np.sum((dst_pts - projected_pts_2)**2, axis=1))
        mean_error = np.mean(error)
        
        # Header
        header_frame = ttk.Frame(self.main_container)
        header_frame.pack(fill=tk.X, pady=10)
        ttk.Label(header_frame, text="Calibration Results", font=("Helvetica", 18, "bold")).pack(side=tk.LEFT)
        
        # Error Status
        err_color = "green" if mean_error < 10 else "red"
        status_text = f"Mean Reprojection Error: {mean_error:.2f} px"
        err_lbl = ttk.Label(header_frame, text=status_text, font=("Helvetica", 14, "bold"), foreground=err_color)
        err_lbl.pack(side=tk.RIGHT)

        # Warning Logic
        if mean_error >= 10:
            warn_frame = ttk.Frame(self.main_container, relief=tk.RAISED, borderwidth=2)
            warn_frame.pack(fill=tk.X, pady=5, padx=5)
            tk.Label(warn_frame, text="⚠ HIGH ERROR WARNING", fg="red", font=("Helvetica", 12, "bold")).pack()
            msg = ("The error is too high (>10px). Suggestions:\n"
                   "1. Ensure the checkerboard is flat and rigid.\n"
                   "2. Ensure good lighting and no motion blur.\n"
                   "3. Cover more angles/positions in the field of view.\n"
                   "4. Check if cameras moved relative to each other.")
            tk.Label(warn_frame, text=msg, justify=tk.LEFT).pack(pady=5)
            
            ttk.Button(warn_frame, text="Discard & Try Again", command=lambda: self.config_screen(self.mode)).pack(pady=5)

        # --- Visual Stats (Notebook for Tabs) ---
        tabs = ttk.Notebook(self.main_container)
        tabs.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Tab 1: Warped View (Visual Check)
        tab_warp = ttk.Frame(tabs)
        tabs.add(tab_warp, text="Warped Perspective")
        
        # Pick the last good capture for visualization
        last_img1 = self.good_captures[-1][0]
        last_img2 = self.good_captures[-1][1]
        
        h, w = last_img2.shape[:2]
        warped_img1 = cv2.warpPerspective(last_img1, self.H, (w, h))
        
        # Blend for comparison
        alpha = 0.5
        blended = cv2.addWeighted(last_img2, alpha, warped_img1, 1 - alpha, 0)
        
        # Display Warped
        self.display_image_in_frame(tab_warp, blended, "Overlay: Cam 1 (Warped) on Cam 2")
        
        # Tab 2: Matched Corners (Side by Side)
        tab_matches = ttk.Frame(tabs)
        tabs.add(tab_matches, text="Matched Corners")
        
        # Draw corners
        vis_img1 = last_img1.copy()
        cv2.drawChessboardCorners(vis_img1, self.board_size, self.good_captures[-1][2], True)
        
        vis_img2 = last_img2.copy()
        cv2.drawChessboardCorners(vis_img2, self.board_size, self.good_captures[-1][3], True)
        
        # Stack horizontal
        vis_img1 = cv2.resize(vis_img1, (640, 480))
        vis_img2 = cv2.resize(vis_img2, (640, 480))
        combined = np.hstack((vis_img1, vis_img2))
        
        self.display_image_in_frame(tab_matches, combined, "Left: Cam 1 | Right: Cam 2")
        
        # Tab 3: Matrix Values
        tab_matrix = ttk.Frame(tabs)
        tabs.add(tab_matrix, text="Homography Matrix")
        
        txt = tk.Text(tab_matrix, height=10, width=50)
        txt.pack(padx=20, pady=20)
        txt.insert(tk.END, str(self.H))
        
        # Save Button
        ttk.Button(self.main_container, text="Save Matrix to File", command=self.save_matrix).pack(pady=10)

    # ---------------- UTILS ----------------
    def display_image_in_frame(self, frame, cv_img, title):
        lbl_title = ttk.Label(frame, text=title)
        lbl_title.pack(pady=5)
        
        # Resize to fit GUI if huge
        h, w = cv_img.shape[:2]
        if w > 1000:
            scale = 1000/w
            cv_img = cv2.resize(cv_img, (int(w*scale), int(h*scale)))
            
        tk_img = self.convert_cv_to_tk(cv_img)
        lbl_img = ttk.Label(frame, image=tk_img)
        lbl_img.image = tk_img # Keep ref
        lbl_img.pack()

    def convert_cv_to_tk(self, cv_img):
        # CV2 is BGR, PIL is RGB
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img_rgb)
        
        # Resize for thumbnails if needed (logic in parent)
        # For live stream resize roughly
        if self.is_streaming:
            pil_img = pil_img.resize((500, 375))
            
        return ImageTk.PhotoImage(pil_img)

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def save_matrix(self):
        filename = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy File", "*.npy"), ("Text File", "*.txt")])
        if filename:
            if filename.endswith('.txt'):
                np.savetxt(filename, self.H)
            else:
                np.save(filename, self.H)
            messagebox.showinfo("Saved", "Homography matrix saved successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = HomographyApp(root)
    root.mainloop()