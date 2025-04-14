import os
import subprocess
import csv
from pathlib import Path

from zip2vid_v0131 import encode_zip_to_video, decode_video_to_zip, check_ffmpeg
import hashlib
import json
from datetime import datetime

# Path to the original ZIP used in encoding
ZIP_PATH = "test2.zip"
RECOVERY_DIR = "simulation_recovered"
os.makedirs(RECOVERY_DIR, exist_ok=True)

# Reference recovery results
RECOVERY_REFERENCE = {
    "bw_2_60_16_0_1744548317": True,
    "bw_2_60_18_0_1744548491": True,
    "bw_4_60_18_0_1744548640": True,
    "bw_4_60_21_0_1744548767": True,
    "bw_4_60_24_0_1744548917": True,
    "bw_8_60_18_0_1744549072": True,
    "bw_8_60_21_0_1744549505": True,
    "bw_8_60_24_0_1744549811": True,
    "4_8_60_18_0_1744550098": True,
    "bw_1_60_18_0_1744551036": False,
    "bw_2_60_24_0_1744551050": True,
    "bw_2_60_28_0_1744551063": False,
    "4_8_60_24_0_1744551076": True,
    "4_12_60_18_0_1744551092": True,
    "4_12_60_21_0_1744551121": True,
    "16_16_60_18_0_1744551150": False,
    "16_20_60_18_0_1744551175": False,
    "16_20_60_21_0_1744551219": False,
    "16_20_60_24_0_1744551261": False,
}


INPUT_DIR = "test_results/videos"
YOUTUBE_DIR = "test_results/manual_downloads"
OUTPUT_DIR = "simulation_outputs"
CSV_PATH = "compression_results.csv"
DECODE_CACHE_FILE = "decode_cache.json"
APPEND_MODE = True  # Set to False to recreate CSV from scratch

os.makedirs(OUTPUT_DIR, exist_ok=True)

crf_values = [12]
bitrates = ["20M"]
cpu_useds = [0] # Map x264 cpu_useds to VP9 CPU usage (0-5, where 0 is slowest/best)

def run_ffmpeg_encode(input_path, output_path, crf, cpu_used, bitrate):

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", "libvpx-vp9",
        "-crf", str(crf),
        "-b:v", bitrate,
        "-cpu-used", str(cpu_used),    # VP9 quality/speed tradeoff
        "-row-mt", "1",           # Enable row-based multithreading
        "-tiles", "4x4",          # Tile encoding for better parallelization
        "-frame-parallel", "1",   # Enable frame parallel processing
        "-pix_fmt", "yuv420p",
        "-auto-alt-ref", "1",     # Enable automatic alternate reference frames
        "-lag-in-frames", "25",   # Allow look-ahead (helps quality)
        "-an"                     # No audio
    ]

    cmd += [str(output_path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_file_size_mb(path):
    return os.path.getsize(path) / 1024 / 1024

def get_ssim_psnr(sim_path, ref_path):
    ssim_cmd = [
        "ffmpeg", "-i", str(sim_path), "-i", str(ref_path),
        "-lavfi", "[0:v][1:v]ssim;[0:v][1:v]psnr",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(ssim_cmd, capture_output=True, text=True, check=True)
        output = result.stderr

        ssim_line = next(line for line in output.splitlines() if "All:" in line and "SSIM" in line)
        psnr_line = next(line for line in output.splitlines() if "average:" in line and "PSNR" in line)

        ssim_val = float(ssim_line.split("All:")[1].split(" ")[0])
        psnr_val = float(psnr_line.split("average:")[1].split()[0])

        return ssim_val, psnr_val
    except Exception as e:
        print(f"SSIM/PSNR failed: {e}")
        return None, None

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def extract_test_params(test_name):
    parts = test_name.split("_")
    try:
        encoding_mode = parts[0]
        block_size = int(parts[1])
        fps = int(parts[2])
        crf = int(parts[3])
        border = int(parts[4])
        return encoding_mode, block_size, fps, crf, border
    except Exception:
        return None, None, None, None, None

def load_decode_cache():
    if os.path.exists(DECODE_CACHE_FILE):
        with open(DECODE_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_decode_cache(cache):
    with open(DECODE_CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def compare_and_record(test_name, encoded_path, youtube_path, crf, cpu_used, bitrate, writer, decode_cache):
    try:
        base_name = test_name.split("_crf")[0]
        
        encoded_size = get_file_size_mb(encoded_path)
        youtube_size = get_file_size_mb(youtube_path)
        diff = abs(encoded_size - youtube_size)
        ratio = encoded_size / youtube_size if youtube_size > 0 else 0

        ssim, psnr = get_ssim_psnr(encoded_path, youtube_path)

        # Check decode cache first
        recovered = False
        match_md5 = False
        if test_name in decode_cache:
            print(f"Using cached decode result for {test_name}")
            cached = decode_cache[test_name]
            recovered = cached['recovered']
            match_md5 = cached['match_md5']
        else:
            # Run recovery check
            encoding_mode, block_size, fps, crf_val, border = extract_test_params(base_name)
            grid_w = 3840 // block_size
            grid_h = 2160 // block_size
            recovered_path = Path(RECOVERY_DIR) / f"{test_name}_recovered.zip"

            decode_result = decode_video_to_zip(
                str(encoded_path), 
                str(recovered_path),
                grid_w, 
                grid_h, 
                border, 
                block_size,
                None,
                encoding_mode
            )

            recovered = decode_result["status"] == "success"

            if recovered:
                original_md5 = calculate_md5(ZIP_PATH)
                recovered_md5 = calculate_md5(recovered_path)
                match_md5 = original_md5 == recovered_md5

            # Cache the result
            decode_cache[test_name] = {
                'recovered': recovered,
                'match_md5': match_md5,
                'timestamp': datetime.now().isoformat()
            }
            save_decode_cache(decode_cache)

        expected_recovery = RECOVERY_REFERENCE.get(base_name)

        writer.writerow({
            "test_name": test_name,
            "crf": crf,
            "cpu_used": cpu_used,
            "bitrate": bitrate or "n/a",
            "encoded_size_MB": round(encoded_size, 2),
            "youtube_size_MB": round(youtube_size, 2),
            "diff_MB": round(diff, 2),
            "size_ratio": round(ratio, 4),
            "ssim": round(ssim, 4) if ssim else "n/a",
            "psnr": round(psnr, 2) if psnr else "n/a",
            "recovery_success": recovered and match_md5,
            "reference_success": expected_recovery if expected_recovery is not None else "unknown",
            "matches_reference": (expected_recovery == (recovered and match_md5)) if expected_recovery is not None else "unknown"
        })
    except Exception as e:
        print(f"Error comparing {test_name}: {e}")

def main():
    input_files = list(Path(INPUT_DIR).glob("*.mp4"))

    # First phase: Encoding
    print("\n=== Phase 1: Encoding Videos ===")
    total_encodes = (
        len(input_files)
        * len(crf_values)
        * len(cpu_useds)
        * len(bitrates)
    )
    completed = 0

    for input_file in input_files:
        base_name = input_file.stem
        youtube_path = Path(YOUTUBE_DIR) / f"{base_name}_youtube_downloaded.mp4"

        if not youtube_path.exists():
            print(f"Skipping {base_name}: no YouTube reference.")
            continue

        for crf in crf_values:
            for cpu_used in cpu_useds:
                for bitrate in bitrates:
                    test_id = f"{base_name}_crf{crf}_{cpu_used}_tune{'none'}_profile{'none'}_br{bitrate}"
                    output_path = Path(OUTPUT_DIR) / f"{test_id}.mp4"

                    # Skip if already encoded
                    if output_path.exists():
                        print(f"[{completed + 1}/{total_encodes}] Skipping existing: {test_id}")
                    else:
                        print(f"[{completed + 1}/{total_encodes}] Encoding: {test_id}")
                        run_ffmpeg_encode(input_file, output_path, crf, cpu_used, bitrate)
                    
                    completed += 1

    # Load decode cache
    decode_cache = load_decode_cache()

    # Second phase: Comparison and Analysis
    print("\n=== Phase 2: Comparing Results ===")
    
    # Check if we should append to existing CSV
    file_mode = "a" if APPEND_MODE and os.path.exists(CSV_PATH) else "w"
    write_header = file_mode == "w" or os.path.getsize(CSV_PATH) == 0
    
    with open(CSV_PATH, mode=file_mode, newline="") as csvfile:
        fieldnames = [
            "test_name", "crf", "cpu_used", "bitrate",
            "encoded_size_MB", "youtube_size_MB", "diff_MB", "size_ratio",
            "ssim", "psnr", "recovery_success", "reference_success", "matches_reference"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        completed = 0

        for input_file in input_files:
            base_name = input_file.stem
            youtube_path = Path(YOUTUBE_DIR) / f"{base_name}_youtube_downloaded.mp4"

            if not youtube_path.exists():
                continue

            for crf in crf_values:
                for cpu_used in cpu_useds:
                    for bitrate in bitrates:
                        test_id = f"{base_name}_crf{crf}_{cpu_used}_tune{'none'}_profile{'none'}_br{bitrate}"
                        output_path = Path(OUTPUT_DIR) / f"{test_id}.mp4"

                        if not output_path.exists():
                            print(f"Warning: Encoded file not found for {test_id}")
                            continue

                        print(f"[{completed + 1}/{total_encodes}] Analyzing: {test_id}")
                        compare_and_record(
                            test_id, output_path, youtube_path,
                            crf, cpu_used, bitrate, writer, decode_cache
                        )
                        completed += 1

    print(f"\nAll results saved to {CSV_PATH}")

if __name__ == "__main__":
    main()
