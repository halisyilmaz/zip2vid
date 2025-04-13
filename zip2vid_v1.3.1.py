import os
import sys
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, StringVar, IntVar, BooleanVar
import multiprocessing as mp
import subprocess
import shutil
import threading
import logging
import hashlib
import struct

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("zip2vid_converter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Z2VConverter")

# === CONFIG ===
DEFAULT_GRID_W, DEFAULT_GRID_H = 960, 540   # grid of blocks in output video
DEFAULT_BLOCK_SIZE = 4                      # block size in pixels
DEFAULT_FPS = 60                            # frames per second for output video
DEFAULT_CRF = 18                            # video quality (lower is better)
DEFAULT_BORDER = 0                          # border width in pixels for blocks
DEFAULT_ENCODING_MODE = "Black & White"     # default encoding mode

# Define a 16-color palette with high contrast to survive YouTube compression
COLOR_PALETTE = [
    (0, 0, 0),        # 0 - black
    (255, 255, 255),  # 1 - white
    (255, 0, 0),      # 2 - red
    (0, 255, 0),      # 3 - green
    (0, 0, 255),      # 4 - blue
    (255, 255, 0),    # 5 - yellow
    (0, 255, 255),    # 6 - cyan
    (255, 0, 255),    # 7 - magenta
    (192, 192, 192),  # 8 - light gray
    (128, 128, 128),  # 9 - gray
    (128, 0, 0),      # 10 - dark red
    (0, 128, 0),      # 11 - dark green
    (0, 0, 128),      # 12 - dark blue
    (128, 128, 0),    # 13 - olive
    (0, 128, 128),    # 14 - teal
    (128, 0, 128),    # 15 - purple
]

# ====== ENCODING FUNCTIONS ======

def encode_chunk_worker(args):
    try:
        (chunk, frame_index, grid_w, grid_h, block_size, border, frame_dir,
         palette_lookup, encoding_mode) = args
        frame_w, frame_h = grid_w * block_size, grid_h * block_size
        data = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        total_blocks = grid_w * grid_h

        if encoding_mode == "bw":
            # Black & White: 1 bit per block
            chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
            bits = np.unpackbits(chunk_arr)[:total_blocks]
            vals = bits
        elif encoding_mode == "4":
            # 4-Gray: 2 bits per block; each byte gives 4 blocks.
            chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
            vals = np.empty(len(chunk_arr) * 4, dtype=np.uint8)
            vals[0::4] = chunk_arr >> 6
            vals[1::4] = (chunk_arr >> 4) & 0x03
            vals[2::4] = (chunk_arr >> 2) & 0x03
            vals[3::4] = chunk_arr & 0x03
            vals = vals[:total_blocks]
        elif encoding_mode == "16":
            # 16-Color: 4 bits per block
            chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
            vals = np.empty(len(chunk_arr) * 2, dtype=np.uint8)
            vals[0::2] = chunk_arr >> 4
            vals[1::2] = chunk_arr & 0x0F
            vals = vals[:total_blocks]
        else:
            raise ValueError("Invalid encoding mode")
        
        idxs = np.arange(total_blocks)
        x_coords = (idxs % grid_w) * block_size
        y_coords = (idxs // grid_w) * block_size
        
        for x, y, val in zip(x_coords, y_coords, vals):
            data[y+border:y+block_size-border, x+border:x+block_size-border] = palette_lookup[val]
        
        img = Image.fromarray(data, mode='RGB')
        frame_path = os.path.join(frame_dir, f"frame_{frame_index:06d}.png")
        img.save(frame_path)
        return frame_index
    except Exception as e:
        logger.error(f"Error in frame processing: {e}")
        return None

def encode_zip_to_video(zip_path, output_video, grid_w, grid_h, block_size, fps, crf, border,
                        encoding_mode, num_processes=None, callback=None):
    try:
        frame_w, frame_h = grid_w * block_size, grid_h * block_size
        total_blocks = grid_w * grid_h
        
        if encoding_mode == "bw":
            chunk_size = total_blocks // 8  # 1 bit per block
            palette_lookup = {0: (0, 0, 0), 1: (255, 255, 255)}
        elif encoding_mode == "4":
            chunk_size = total_blocks // 4  # 2 bits per block
            palette_lookup = {
                0: (0, 0, 0),         # black
                1: (85, 85, 85),      # dark gray
                2: (170, 170, 170),   # light gray
                3: (255, 255, 255)    # white
            }
        elif encoding_mode == "16":
            chunk_size = total_blocks // 2  # 4 bits per block
            palette_lookup = {i: COLOR_PALETTE[i] for i in range(16)}
        else:
            raise ValueError("Invalid encoding mode")
        
        frame_dir = "frames"
        os.makedirs(frame_dir, exist_ok=True)
        
        with open(zip_path, 'rb') as f:
            original_data = f.read()
        hash_digest = hashlib.sha256(original_data).digest()
        final_data = struct.pack(">I", len(original_data)) + original_data + hash_digest
        
        chunks = [final_data[i:i + chunk_size] for i in range(0, len(final_data), chunk_size)]
        total_chunks = len(chunks)
        
        # Create arguments for worker function; include encoding_mode.
        args_list = [(chunk, i, grid_w, grid_h, block_size, border, frame_dir, palette_lookup, encoding_mode)
                    for i, chunk in enumerate(chunks)]
        
        completed = 0
        with mp.Pool(processes=num_processes or max(1, mp.cpu_count() - 1)) as pool:
            for _ in pool.imap_unordered(encode_chunk_worker, args_list):
                completed += 1
                if callback:
                    progress = (completed / total_chunks) * 100
                    callback(progress, f"Processed {completed}/{total_chunks} frames")
        
        if total_chunks < fps:
            black_frame = Image.new("RGB", (frame_w, frame_h), (0, 0, 0))
            for i in range(total_chunks, fps):
                black_frame.save(os.path.join(frame_dir, f"frame_{i:06d}.png"))
                if callback:
                    pad_progress = ((i - total_blocks + 1) / (fps - total_chunks) * 10) + 90
                    callback(pad_progress, f"Adding padding frame {i - total_chunks + 1}/{fps - total_chunks}")
        
        if callback:
            callback(95, "Creating video with ffmpeg...")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "auto",
            "-framerate", str(fps),
            "-i", f"{frame_dir}/frame_%06d.png",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
            output_video
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        shutil.rmtree(frame_dir, ignore_errors=True)
        input_size = os.path.getsize(zip_path)
        output_size = os.path.getsize(output_video)
        ratio = output_size / input_size
        
        logger.info(f"[OK] Video created successfully: {output_video}")
        logger.info(f"[OK] Input size: {input_size/1024:.2f}KB, Output size: {output_size/1024:.2f}KB, Ratio: {ratio:.2f}x")
        if callback:
            callback(100, "Complete!")
        return {"status": "success", "input_size": input_size, "output_size": output_size,
                "ratio": ratio, "total_frames": total_chunks}
    
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg error: {e.stderr.decode('utf-8') if e.stderr else str(e)}"
        logger.error(error_msg)
        if callback:
            callback(-1, error_msg)
        return {"status": "error", "message": error_msg}
    
    except Exception as e:
        error_msg = f"Encoding error: {str(e)}"
        logger.error(error_msg)
        if callback:
            callback(-1, error_msg)
        return {"status": "error", "message": error_msg}

# ====== DECODING FUNCTIONS ======

def decode_frame(args, encoding_mode):
    try:
        frame, palette_array, grid_w, grid_h, border, block_size = args
        # Reshape frame to (grid_h, block_size, grid_w, block_size, 3)
        blocks = frame.reshape(grid_h, block_size, grid_w, block_size, 3)
        if border > 0:
            blocks = blocks[:, border:-border, :, border:-border, :]
        # Compute block means for all blocks at once
        block_means = blocks.mean(axis=(1,3))  # shape: (grid_h, grid_w, 3)
        block_means = block_means.reshape(-1, 3)  # shape: (total_blocks, 3)
        
        if encoding_mode == "bw":
            # For Black & White, just convert the mean intensity to a bit.
            # We use the average intensity across channels.
            intensities = block_means.mean(axis=1)
            bits = (intensities > 127).astype(np.uint8)
            # Pack 8 bits into a byte
            recovered = bytearray()
            for i in range(0, len(bits), 8):
                b = 0
                for bit in bits[i:i+8]:
                    b = (b << 1) | int(bit)
                recovered.append(b)
            return recovered
        
        elif encoding_mode == "4":
            # For 4-Gray mode, use a grayscale palette.
            # Use vectorized distance computation.
            palette = np.array([(0, 0, 0), (85, 85, 85), (170, 170, 170), (255, 255, 255)], dtype=np.float32)
            diff = block_means[:, np.newaxis, :] - palette[np.newaxis, :, :]
            distances = np.sum(diff**2, axis=2)  # shape: (total_blocks, 4)
            indices = np.argmin(distances, axis=1).astype(np.uint8)  # each value in 0..3
            # Now pack every 4 two-bit values into a byte.
            recovered = bytearray()
            for i in range(0, len(indices), 4):
                b = 0
                for val in indices[i:i+4]:
                    b = (b << 2) | int(val)
                recovered.append(b)
            return recovered
        
        elif encoding_mode == "16":
            # For 16-Color mode, use the optimized method (can be similarly vectorized)
            # Reverse average color from BGR to RGB
            block_means = block_means[:, ::-1]
            # Use vectorized computation
            palette = np.array(palette_array, dtype=np.float32)
            diff = block_means[:, np.newaxis, :] - palette[np.newaxis, :, :]
            distances = np.sum(diff**2, axis=2)
            indices = np.argmin(distances, axis=1).astype(np.uint8)
            # Pack every 2 4-bit values into one byte.
            recovered = bytearray()
            for i in range(0, len(indices), 2):
                high = indices[i]
                low = indices[i+1] if i+1 < len(indices) else 0
                recovered.append((high << 4) | low)
            return recovered
        
        else:
            raise ValueError("Invalid encoding mode")
    
    except Exception as e:
        logger.error(f"Error in frame decoding: {e}")
        return bytearray()
    
def decode_video_to_zip(video_path, output_zip, grid_w, grid_h, border, block_size, callback=None, encoding_mode="bw"):
    try:
        frame_w, frame_h = grid_w * block_size, grid_h * block_size
        frame_size = frame_w * frame_h * 3  # bgr24 = 3 bytes per pixel
        palette_array = np.array(COLOR_PALETTE, dtype=np.float32)
        
        # Get total frame count for progress reporting
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", 
            "-count_packets", "-show_entries", "stream=nb_read_packets", 
            "-of", "csv=p=0", video_path
        ]
        
        try:
            total_frames = int(subprocess.check_output(probe_cmd).decode().strip())
        except:
            total_frames = 100  # Default estimate if probe fails
            
        if callback:
            callback(5, "Extracting frames from video...")
        
        # Use ffmpeg to extract raw frames (BGR24)
        cmd = [
            "ffmpeg", 
            "-hwaccel", "auto",
            "-i", video_path,
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo", "-"
        ]
        
        recovered_data = []
        processed_frames = 0

        with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8, creationflags=subprocess.CREATE_NO_WINDOW) as pipe:
            with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
                results = []
                while True:
                    raw_frame = pipe.stdout.read(frame_size)
                    if not raw_frame or len(raw_frame) != frame_size:
                        break
                    
                    frame = np.frombuffer(raw_frame, np.uint8)
                    args = (frame, palette_array, grid_w, grid_h, border, block_size)
                    results.append(pool.apply_async(decode_frame, (args, encoding_mode)))
                    
                    processed_frames += 1
                    if callback and processed_frames % 10 == 0:
                        progress = min(90, (processed_frames / total_frames) * 90)
                        callback(progress, f"Processing frame {processed_frames}/{total_frames}")
                
                if callback:
                    callback(95, "Assembling ZIP file...")
                
                for res in results:
                    recovered_data.append(res.get())
            
        recovered_bytes = b''.join(recovered_data)
        
        if len(recovered_bytes) < 4 + 32:
            return {"status": "error", "message": "Recovered data too short"}

        zip_len = struct.unpack(">I", recovered_bytes[:4])[0]
        file_data = recovered_bytes[4:4 + zip_len]
        expected_hash = recovered_bytes[4 + zip_len: 4 + zip_len + 32]
        actual_hash = hashlib.sha256(file_data).digest()

        if actual_hash != expected_hash:
            logger.error("SHA256 mismatch! Data may be corrupted.")
            return {"status": "error", "message": "SHA256 hash mismatch. Data may be corrupted."}

        with open(output_zip, 'wb') as f:
            f.write(file_data)
        
        if callback:
            callback(100, "Complete!")
        
        logger.info(f"[OK] ZIP file restored: {output_zip}")
        return {"status": "success", "output_path": output_zip, "frames_processed": processed_frames}
    
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg error: {e.stderr.decode('utf-8') if e.stderr else str(e)}"
        logger.error(error_msg)
        if callback:
            callback(-1, error_msg)
        return {"status": "error", "message": error_msg}
    
    except Exception as e:
        error_msg = f"Decoding error: {str(e)}"
        logger.error(error_msg)
        if callback:
            callback(-1, error_msg)
        return {"status": "error", "message": error_msg}

# ====== UTILITY FUNCTIONS ======

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def calculate_stats(grid_w, grid_h, block_size):
    frame_w = grid_w * block_size
    frame_h = grid_h * block_size
    data_per_frame = (grid_w * grid_h) // 2  # bytes per frame (2 blocks per byte)
    
    return {
        "resolution": f"{frame_w}x{frame_h}",
        "data_per_frame": data_per_frame,
        "blocks": grid_w * grid_h
    }

# ====== GUI CLASS ======

class Zip2VidVideoConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Zip2Vid Converter")
        self.root.minsize(800, 600)
        
        # Check FFmpeg availability
        if not check_ffmpeg():
            messagebox.showerror("Error", "FFmpeg not found! Please install FFmpeg and add it to your PATH.")
            self.root.quit()
            return
        
        # Set up styles
        self.style = ttk.Style()
        self.style.configure("TNotebook", padding=10)
        self.style.configure("TButton", padding=5)
        self.style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        
        # Create main tab control
        self.tab_control = ttk.Notebook(root)
        
        # Create tabs
        self.encode_tab = ttk.Frame(self.tab_control)
        self.decode_tab = ttk.Frame(self.tab_control)
        self.settings_tab = ttk.Frame(self.tab_control)
        self.about_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.encode_tab, text="Encode ZIP to Video")
        self.tab_control.add(self.decode_tab, text="Decode Video to ZIP")
        self.tab_control.add(self.settings_tab, text="Settings")
        self.tab_control.add(self.about_tab, text="About")
        self.tab_control.pack(expand=1, fill="both")
        
        # Initialize variables
        self.setup_variables()
        
        # Create UI tabs
        self.setup_encode_tab()
        self.setup_decode_tab()
        self.setup_settings_tab()
        self.setup_about_tab()
        
        # Set up status bar
        self.status_var = StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_bar.pack(side="bottom", fill="x")
        
        # Update stats
        self.update_stats()
    
    def setup_variables(self):
        # File paths
        self.zip_path = StringVar()
        self.video_output_path = StringVar()
        self.video_input_path = StringVar()
        self.zip_output_path = StringVar()
        
        # Settings
        self.grid_w = IntVar(value=DEFAULT_GRID_W)
        self.grid_h = IntVar(value=DEFAULT_GRID_H)
        self.block_size = IntVar(value=DEFAULT_BLOCK_SIZE)
        self.fps = IntVar(value=DEFAULT_FPS)
        self.crf = IntVar(value=DEFAULT_CRF)
        self.border = IntVar(value=DEFAULT_BORDER)
        self.use_max_threads = BooleanVar(value=True)
        self.num_threads = IntVar(value=max(1, mp.cpu_count() - 1))
        
        # New encoding mode variable: options "Black & White", "16-Color"
        self.encoding_mode = StringVar(value=DEFAULT_ENCODING_MODE)
        
        # Attach trace on block_size to update stats and output filename.
        self.block_size.trace_add("write", lambda *args: (self.update_stats(), self.update_video_output_filename()))
        # Attach trace on encoding mode to update output filename.
        self.encoding_mode.trace_add("write", lambda *args: self.update_video_output_filename())
        
        # Status
        self.encode_status = StringVar(value="Ready to encode")
        self.decode_status = StringVar(value="Ready to decode")
        
        # Statistics
        self.resolution = StringVar()
        self.data_per_frame = StringVar()
        self.blocks = StringVar()
    
    def setup_encode_tab(self):
        frame = self.encode_tab
        frame.columnconfigure(1, weight=1)
        
        # Heading
        ttk.Label(frame, text="Encode ZIP File to Video", style="Header.TLabel").grid(
            row=0, column=0, columnspan=3, pady=10, sticky="w")
        
        # Input ZIP file
        ttk.Label(frame, text="ZIP File:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.zip_path, width=50).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_zip_input).grid(row=1, column=2, padx=5, pady=5)
        
        # Output video file
        ttk.Label(frame, text="Output Video:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.video_output_path, width=50).grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_video_output).grid(row=2, column=2, padx=5, pady=5)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(frame, text="Encoding Stats")
        stats_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        stats_frame.columnconfigure(1, weight=1)
        
        ttk.Label(stats_frame, text="Resolution:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(stats_frame, textvariable=self.resolution).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(stats_frame, text="Data per Frame:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(stats_frame, textvariable=self.data_per_frame).grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(stats_frame, text="Total Blocks:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(stats_frame, textvariable=self.blocks).grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        # Progress bar
        ttk.Label(frame, text="Progress:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.encode_progress = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
        self.encode_progress.grid(row=4, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Status
        ttk.Label(frame, textvariable=self.encode_status).grid(row=5, column=0, columnspan=3, padx=5, pady=5)
        
        # Encode button
        ttk.Button(frame, text="Encode ZIP to Video", command=self.start_encode).grid(
            row=6, column=1, pady=20)
    
    def setup_decode_tab(self):
        frame = self.decode_tab
        frame.columnconfigure(1, weight=1)
        
        # Heading
        ttk.Label(frame, text="Decode Video to ZIP File", style="Header.TLabel").grid(
            row=0, column=0, columnspan=3, pady=10, sticky="w")
        
        # Input video file
        ttk.Label(frame, text="Video File:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.video_input_path, width=50).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_video_input).grid(row=1, column=2, padx=5, pady=5)
        
        # Output ZIP file
        ttk.Label(frame, text="Output ZIP:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.zip_output_path, width=50).grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_zip_output).grid(row=2, column=2, padx=5, pady=5)
        
        # Progress bar
        ttk.Label(frame, text="Progress:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.decode_progress = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
        self.decode_progress.grid(row=3, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Status
        ttk.Label(frame, textvariable=self.decode_status).grid(row=4, column=0, columnspan=3, padx=5, pady=5)
        
        # Decode button
        ttk.Button(frame, text="Decode Video to ZIP", command=self.start_decode).grid(
            row=5, column=1, pady=20)
    
    def setup_settings_tab(self):
        frame = self.settings_tab
        frame.columnconfigure(1, weight=1)
        
        # Heading
        ttk.Label(frame, text="Converter Settings", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, pady=10, sticky="w")
        
        # Add Encoding Mode setting
        ttk.Label(frame, text="Encoding Mode:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        mode_options = ["Black & White", "4-Gray", "16-Color"]
        ttk.Combobox(frame, textvariable=self.encoding_mode, values=mode_options, 
                    state="readonly").grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(frame, text="B&W(recommended) - best ratio(~3.5) and fastest").grid(
            row=1, column=2, sticky="w", padx=5, pady=5)
        
        # Block size
        ttk.Label(frame, text="Block Size (px):").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        ttk.Spinbox(frame, from_=4, to=40, increment=2, textvariable=self.block_size,
                   command=self.update_stats).grid(row=2, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(frame, text="B&W: 4(recommended) / 4-Gray: 8 / 16-Color: 20").grid(
            row=2, column=2, sticky="w", padx=5, pady=5)
        
        # Grid dimensions
        ttk.Label(frame, text="Grid Width:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        ttk.Spinbox(frame, from_=16, to=512, increment=8, textvariable=self.grid_w, 
                   command=self.update_stats).grid(row=3, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(frame, text="4K(recommended): 3860/block_size").grid(
            row=3, column=2, sticky="w", padx=5, pady=5)
        
        ttk.Label(frame, text="Grid Height:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        ttk.Spinbox(frame, from_=16, to=512, increment=8, textvariable=self.grid_h,
                   command=self.update_stats).grid(row=4, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(frame, text="4K(recommended): 2160/block_size").grid(
            row=4, column=2, sticky="w", padx=5, pady=5)

        # FPS
        ttk.Label(frame, text="Video FPS:").grid(row=5, column=0, sticky="e", padx=5, pady=5)
        ttk.Spinbox(frame, from_=15, to=120, increment=5, textvariable=self.fps).grid(
            row=5, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(frame, text="60(recommended)").grid(
            row=5, column=2, sticky="w", padx=5, pady=5)
        
        # Video quality (CRF)
        ttk.Label(frame, text="Video Quality (CRF):").grid(row=6, column=0, sticky="e", padx=5, pady=5)
        ttk.Spinbox(frame, from_=0, to=51, increment=1, textvariable=self.crf).grid(
            row=6, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(frame, text="(Lower is better quality, 18-24 recommended)").grid(
            row=6, column=2, sticky="w", padx=5, pady=5)
             
        # Border for Color Blocks (px)
        ttk.Label(frame, text="Block Border Width (px):").grid(row=7, column=0, sticky="e", padx=5, pady=5)
        ttk.Spinbox(frame, from_=0, to=51, increment=1, textvariable=self.border).grid(
            row=7, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(frame, text="EXPERIMENTAL! Adds black border between blocks").grid(
            row=7, column=2, sticky="w", padx=5, pady=5)
                
        # Threading options
        ttk.Label(frame, text="CPU Threads:").grid(row=8, column=0, sticky="e", padx=5, pady=5)
        
        thread_frame = ttk.Frame(frame)
        thread_frame.grid(row=8, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Radiobutton(thread_frame, text="Use maximum", variable=self.use_max_threads, 
                       value=True, command=self.toggle_threads).pack(side="left")
        ttk.Radiobutton(thread_frame, text="Specify:", variable=self.use_max_threads, 
                       value=False, command=self.toggle_threads).pack(side="left")
        
        self.thread_spinbox = ttk.Spinbox(thread_frame, from_=1, to=mp.cpu_count(), 
                                        increment=1, width=5, textvariable=self.num_threads)
        self.thread_spinbox.pack(side="left", padx=5)
        
        if self.use_max_threads.get():
            self.thread_spinbox.configure(state="disabled")
    
        
        # Reset button
        ttk.Button(frame, text="Reset to Defaults", command=self.reset_settings).grid(
            row=9, column=1, sticky="w", padx=5, pady=20)
    
    def setup_about_tab(self):
        frame = self.about_tab
        
        # Heading
        ttk.Label(frame, text="Zip2Vid Video Converter", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Description
        description = (
            "This tool allows you to convert ZIP files to video format and vice versa, "
            "enabling data storage and transfer via video platforms like YouTube.\n\n"
            "The application uses a color grid encoding technique that is resistant to "
            "video compression, making it suitable for transferring data via video sharing platforms."
        )
        ttk.Label(frame, text=description, wraplength=600, justify="center").pack(pady=10)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(frame, text="Instructions")
        instructions_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        instructions = (
            "1. Encode: Select a ZIP file and specify an output video path.\n"
            "2. Upload: Upload the generated video to YouTube or other platform.\n"
            "3. Download: Download the video from the platform.\n"
            "4. Decode: Use the decode tab to convert the video back to the original ZIP file.\n\n"
            "For best results with YouTube:\n"
            "- Use 4 block size in black & white mode \n"
            "- Use 4K(3840x2160) and 60FPS. \n"
            "- Wait for YouTube processing to complete before downloading\n"
            "- Download at 4K 60FPS quality"
        )
        ttk.Label(instructions_frame, text=instructions, justify="left").pack(padx=10, pady=10, anchor="w")
        
        # System info
        sys_frame = ttk.LabelFrame(frame, text="System Information")
        sys_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        sys_info = (
            f"CPU Cores: {mp.cpu_count()}\n"
            f"Python Version: {sys.version.split()[0]}\n"
            f"Default Resolution: {DEFAULT_GRID_W * DEFAULT_BLOCK_SIZE}x{DEFAULT_GRID_H * DEFAULT_BLOCK_SIZE}"
        )
        ttk.Label(sys_frame, text=sys_info, justify="left").pack(padx=10, pady=10, anchor="w")
        
        # Credits
        ttk.Label(frame, text="© 2025 Zip2Vid Video Converter", font=("Arial", 10)).pack(side="bottom", pady=10)
    
    # ====== Event Handlers ======
    def update_video_output_filename(self):
        if not self.zip_path.get():
            return
        try:
            bs = self.block_size.get()
        except tk.TclError:
            bs = DEFAULT_BLOCK_SIZE

        base = os.path.splitext(self.zip_path.get())[0]
        mode_ui = self.encoding_mode.get()
        if mode_ui == "Black & White":
            mode_code = "BW"
        elif mode_ui == "4-Gray":
            mode_code = "4G"
        elif mode_ui == "16-Color":
            mode_code = "16C"
        else:
            mode_code = "BW"
        output_path = f"{base}_{mode_code}_{bs}block.mp4"
        self.video_output_path.set(output_path)
    
    def browse_zip_input(self):
        """Browse for input ZIP file."""
        path = filedialog.askopenfilename(filetypes=[("ZIP Files", "*.zip"), ("All Files", "*.*")])
        if path:
            self.zip_path.set(path)
            self.update_video_output_filename()
    
    def browse_video_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")]
        )
        if path:
            self.video_output_path.set(path)
    
    def browse_video_input(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 Video", "*.mp4"), ("All Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*")])
        if path:
            self.video_input_path.set(path)
            # Auto-suggest output ZIP file name every time a video is selected.
            base = os.path.splitext(path)[0]
            output_path = f"{base}_restored.zip"
            self.zip_output_path.set(output_path)

    def browse_zip_output(self):
        """Browse for output ZIP file."""
        path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("ZIP Files", "*.zip"), ("All Files", "*.*")]
        )
        if path:
            self.zip_output_path.set(path)
    
    def toggle_threads(self):
        """Toggle thread count spinbox state."""
        if self.use_max_threads.get():
            self.thread_spinbox.configure(state="disabled")
        else:
            self.thread_spinbox.configure(state="normal")
    
    def update_stats(self, *args):
        try:
            block_size = self.block_size.get()
        except (tk.TclError, ValueError):
            return  # Exit if the block_size is invalid or empty

        # Calculate grid dimensions to target 3840x2160 output resolution.
        new_grid_w = 3840 // block_size
        new_grid_h = 2160 // block_size
        self.grid_w.set(new_grid_w)
        self.grid_h.set(new_grid_h)
        
        stats = calculate_stats(new_grid_w, new_grid_h, block_size)
        self.resolution.set(f"{new_grid_w * block_size}x{new_grid_h * block_size}")
        self.data_per_frame.set(f"{stats['data_per_frame']:,} bytes")
        self.blocks.set(f"{stats['blocks']:,}")
    
    def reset_settings(self):
        """Reset all settings to defaults."""
        self.grid_w.set(DEFAULT_GRID_W)
        self.grid_h.set(DEFAULT_GRID_H)
        self.block_size.set(DEFAULT_BLOCK_SIZE)
        self.fps.set(DEFAULT_FPS)
        self.crf.set(DEFAULT_CRF)
        self.border.set(DEFAULT_BORDER)
        self.update_stats()
    
    def update_encode_progress(self, progress, status_text=None):
        """Update the encode progress bar and status text."""
        if not self.root:
            return
            
        if progress < 0:  # Error state
            self.encode_status.set(status_text or "Error occurred")
            return
            
        self.encode_progress["value"] = progress
        if status_text:
            self.encode_status.set(status_text)
        self.root.update_idletasks()
    
    def update_decode_progress(self, progress, status_text=None):
        """Update the decode progress bar and status text."""
        if not self.root:
            return
            
        if progress < 0:  # Error state
            self.decode_status.set(status_text or "Error occurred")
            return
            
        self.decode_progress["value"] = progress
        if status_text:
            self.decode_status.set(status_text)
        self.root.update_idletasks()
    
    def start_encode(self):
        # (Validate input file paths as before)
        if not self.zip_path.get():
            messagebox.showerror("Error", "Please select an input ZIP file.")
            return
        if not self.video_output_path.get():
            messagebox.showerror("Error", "Please specify an output video path.")
            return
        if not os.path.exists(self.zip_path.get()):
            messagebox.showerror("Error", "Input ZIP file does not exist.")
            return
        
        self.encode_progress["value"] = 0
        self.encode_status.set("Starting encoding process...")
        
        threads = None if self.use_max_threads.get() else self.num_threads.get()
        output_dir = os.path.dirname(self.video_output_path.get())
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Map UI option to internal encoding mode string.
        mode_ui = self.encoding_mode.get()
        if mode_ui == "Black & White":
            mode = "bw"
        elif mode_ui == "4-Gray":
            mode = "4"
        elif mode_ui == "16-Color":
            mode = "16"
        else:
            mode = "bw"
        
        def run_encode():
            try:
                result = encode_zip_to_video(
                    self.zip_path.get(),
                    self.video_output_path.get(),
                    self.grid_w.get(),
                    self.grid_h.get(),
                    self.block_size.get(),
                    self.fps.get(),
                    self.crf.get(),
                    self.border.get(),
                    mode,
                    threads,
                    self.update_encode_progress
                )
                if result["status"] == "success":
                    self.update_encode_progress(100, "Encoding completed successfully!")
                    self.status_var.set(f"Encoded ZIP ({result['input_size']/1024:.1f} KB) to video ({result['output_size']/1024:.1f} KB) - Ratio: {result['ratio']:.2f}x")
                    messagebox.showinfo("Success", f"ZIP file encoded to video successfully!\n\nInput size: {result['input_size']/1024:.1f} KB\nOutput size: {result['output_size']/1024:.1f} KB\nRatio: {result['ratio']:.2f}x")
                else:
                    self.update_encode_progress(-1, f"Error: {result['message']}")
                    messagebox.showerror("Error", f"Encoding failed: {result['message']}")
            except Exception as e:
                self.update_encode_progress(-1, f"Error: {str(e)}")
                messagebox.showerror("Error", f"Encoding failed: {str(e)}")
        
        threading.Thread(target=run_encode, daemon=True).start()

    def start_decode(self):
        if not self.video_input_path.get():
            messagebox.showerror("Error", "Please select an input video file.")
            return
        if not self.zip_output_path.get():
            messagebox.showerror("Error", "Please specify an output ZIP path.")
            return
        if not os.path.exists(self.video_input_path.get()):
            messagebox.showerror("Error", "Input video file does not exist.")
            return
        
        self.decode_progress["value"] = 0
        self.decode_status.set("Starting decoding process...")
        output_dir = os.path.dirname(self.zip_output_path.get())
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        mode_ui = self.encoding_mode.get()
        if mode_ui == "Black & White":
            mode = "bw"
        elif mode_ui == "4-Gray":
            mode = "4"
        elif mode_ui == "16-Color":
            mode = "16"
        else:
            mode = "bw"
        
        def run_decode():
            try:
                result = decode_video_to_zip(
                    self.video_input_path.get(),
                    self.zip_output_path.get(),
                    self.grid_w.get(),
                    self.grid_h.get(),
                    self.border.get(),
                    self.block_size.get(),
                    self.update_decode_progress,
                    mode  # Pass mode into decoding functions as needed.
                )
                if result["status"] == "success":
                    self.update_decode_progress(100, "Decoding completed successfully!")
                    self.status_var.set(f"Decoded video to ZIP - {result['frames_processed']} frames processed")
                    messagebox.showinfo("Success", f"Video decoded to ZIP file successfully!\n\nFrames processed: {result['frames_processed']}")
                else:
                    self.update_decode_progress(-1, f"Error: {result['message']}")
                    messagebox.showerror("Error", f"Decoding failed: {result['message']}")
            except Exception as e:
                self.update_decode_progress(-1, f"Error: {str(e)}")
                messagebox.showerror("Error", f"Decoding failed: {str(e)}")
        
        threading.Thread(target=run_decode, daemon=True).start()

# ====== MAIN APPLICATION ======

def main():
    """Main application entry point."""
    # Configure better DPI handling for Windows
    if os.name == 'nt':
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
    
    # Set up root window
    root = tk.Tk()
    root.geometry("850x650")
    root.resizable(True, True)   
     
    # Create application
    app = Zip2VidVideoConverterGUI(root)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    mp.freeze_support()  # Required for frozen executables on Windows
    main()

