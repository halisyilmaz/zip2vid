#!/usr/bin/env python3
import os
import sys
import time
import json
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Import the necessary functions from your module
# Adjust this import to match your module structure
sys.path.append('.')  # Add current directory to path
from zip2vid_v0131 import encode_zip_to_video, decode_video_to_zip, check_ffmpeg

class YouTubeCompressionTester:
    def __init__(self, test_zip_path, output_dir="test_results", manual_mode=False):
        """
        Initialize the YouTube compression tester
        
        Args:
            test_zip_path: Path to the ZIP file to use for testing
            output_dir: Directory to store test results
            manual_mode: If True, will wait for user to manually upload/download videos
        """
        self.test_zip_path = test_zip_path
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "videos"
        self.youtube_sim_dir = self.output_dir / "youtube_simulated"
        self.manual_downloads_dir = self.output_dir / "manual_downloads"
        self.recovered_dir = self.output_dir / "recovered"
        self.results_file = self.output_dir / "test_results.csv"
        self.summary_file = self.output_dir / "test_summary.json"
        self.manual_mode = manual_mode
        self.batch_files = []  # To track files for batch processing
        
        # Create all necessary directories
        for dir_path in [self.output_dir, self.results_dir, self.youtube_sim_dir, 
                         self.manual_downloads_dir, self.recovered_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Check if ffmpeg is available
        if not check_ffmpeg():
            print("Error: FFmpeg not found! Please install FFmpeg and add it to your PATH.")
            sys.exit(1)
        
        # Initialize results dataframe
        if os.path.exists(self.results_file):
            self.results_df = pd.read_csv(self.results_file)
            print(f"Loaded existing results file with {len(self.results_df)} test cases")
        else:
            self.results_df = pd.DataFrame(columns=[
                'test_id', 'encoding_mode', 'block_size', 'grid_w', 'grid_h', 
                'fps', 'crf', 'border', 'input_size_kb', 'output_size_kb', 
                'youtube_size_kb', 'ratio_before_youtube', 'ratio_after_youtube',
                'recovery_success', 'checksums_match', 'timestamp'
            ])
    
    def simulate_youtube_compression(self, input_video, output_video):
        """Simulate YouTube compression using FFmpeg with more forgiving settings"""
        # Adjusted settings based on real YouTube behavior
        cpu_used = "0"           # Less aggressive preset
        crf = "12"                 # Higher quality (lower CRF)
        bitrate = "20M"            # Higher bitrate for 4K


        # Single-pass encoding (YouTube likely uses something similar)
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-c:v", "libvpx-vp9",
            "-crf", crf,
            "-b:v", bitrate,
            "-cpu-used", cpu_used,    # VP9 quality/speed tradeoff
            "-row-mt", "1",           # Enable row-based multithreading
            "-tiles", "4x4",          # Tile encoding for better parallelization
            "-frame-parallel", "1",   # Enable frame parallel processing
            "-pix_fmt", "yuv420p",
            "-auto-alt-ref", "1",     # Enable automatic alternate reference frames
            "-lag-in-frames", "25",   # Allow look-ahead (helps quality)
            "-an",                     # No audio
            output_video
        ]   

        print("Running compression simulation...")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_video
    
    def calculate_md5(self, file_path):
        """Calculate MD5 hash of a file"""
        import hashlib
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def encode_test_video(self, test_id, encoding_mode, block_size, grid_w, grid_h, 
                          fps, crf, border, callback=None):
        """
        Encode a test video and return information about it
        
        Args:
            All parameters needed for encoding
            
        Returns:
            Dictionary with video information
        """
        video_path = self.results_dir / f"{test_id}.mp4"
        manual_download_path = self.manual_downloads_dir / f"{test_id}_youtube_downloaded.mp4"

        # Check if video already exists
        if video_path.exists():
            print(f"Using existing encoded video: {video_path}")
            input_size = os.path.getsize(self.test_zip_path)
            output_size = os.path.getsize(video_path)
            
            # Return existing video info
            if self.manual_mode:
                return {
                    'status': 'encoded',
                    'test_id': test_id,
                    'video_path': video_path,
                    'input_size': input_size,
                    'output_size': output_size
                }
            
            # For non-manual mode, check/create YouTube simulation
            youtube_path = self.youtube_sim_dir / f"{test_id}_youtube.mp4"
            if not youtube_path.exists():
                print(f"Simulating YouTube compression...")
                self.simulate_youtube_compression(str(video_path), str(youtube_path))
            else:
                print(f"Using existing YouTube simulation: {youtube_path}")
                
            youtube_size = os.path.getsize(youtube_path)
            
            return {
                'status': 'compressed',
                'test_id': test_id,
                'video_path': video_path,
                'youtube_path': youtube_path,
                'youtube_size': youtube_size,
                'input_size': input_size,
                'output_size': output_size
            }
        
        print(f"\n{'='*80}\nEncoding: Mode={encoding_mode}, Block Size={block_size}, "
              f"Grid={grid_w}x{grid_h}, FPS={fps}, CRF={crf}, Border={border}\n{'='*80}")
        
        # Map encoding_mode string to internal format
        if encoding_mode == "Black & White":
            enc_mode = "bw"
        elif encoding_mode == "4-Gray":
            enc_mode = "4"
        elif encoding_mode == "16-Color":
            enc_mode = "16"
        else:
            enc_mode = encoding_mode  # Assume it's already in correct format
        
        # Step 1: Encode ZIP to video
        print(f"Encoding ZIP to video...")
        encode_result = encode_zip_to_video(
            self.test_zip_path,
            str(video_path),
            grid_w,
            grid_h,
            block_size,
            fps,
            crf,
            border,
            enc_mode,
            callback=callback
        )
        
        if encode_result["status"] != "success":
            print(f"Encoding failed: {encode_result['message']}")
            return None
        
        input_size = os.path.getsize(self.test_zip_path)
        output_size = os.path.getsize(video_path)
        
        # For manual mode, add to batch files
        if self.manual_mode:
            self.batch_files.append({
                'test_id': test_id,
                'encoding_mode': encoding_mode,
                'enc_mode': enc_mode,
                'block_size': block_size,
                'grid_w': grid_w,
                'grid_h': grid_h,
                'fps': fps,
                'crf': crf,
                'border': border,
                'input_size': input_size,
                'output_size': output_size,
                'video_path': video_path,
                'manual_download_path': manual_download_path
            })
            
            return {
                'status': 'encoded',
                'test_id': test_id,
                'video_path': video_path
            }
        
        # For non-manual mode, simulate YouTube compression
        youtube_path = self.youtube_sim_dir / f"{test_id}_youtube.mp4"
        print(f"Simulating YouTube compression...")
        self.simulate_youtube_compression(str(video_path), str(youtube_path))
        youtube_size = os.path.getsize(youtube_path)
        
        return {
            'status': 'compressed',
            'test_id': test_id,
            'video_path': video_path,
            'youtube_path': youtube_path,
            'youtube_size': youtube_size,
            'input_size': input_size,
            'output_size': output_size
        }
    
    def process_youtube_videos(self, callback=None):
        """
        Process all YouTube downloaded videos in batch mode
        
        Returns:
            List of results
        """
        results = []
        
        print(f"\n{'='*80}")
        print(f"Processing {len(self.batch_files)} downloaded YouTube videos")
        print(f"{'='*80}")
        
        for idx, file_info in enumerate(self.batch_files):
            test_id = file_info['test_id']
            recovered_path = self.recovered_dir / f"{test_id}_recovered.zip"
            manual_download_path = file_info['manual_download_path']
            
            if not os.path.exists(manual_download_path):
                print(f"Warning: Downloaded video not found for test {test_id}. Skipping.")
                continue
            
            # Get YouTube file size
            youtube_size = os.path.getsize(manual_download_path)
            
            # Decode YouTube video back to ZIP
            print(f"\nDecoding YouTube video {idx+1}/{len(self.batch_files)}: {test_id}")
            decode_result = decode_video_to_zip(
                str(manual_download_path),
                str(recovered_path),
                file_info['grid_w'],
                file_info['grid_h'],
                file_info['border'],
                file_info['block_size'],
                callback=callback,
                encoding_mode=file_info['enc_mode']
            )
            
            recovery_success = decode_result["status"] == "success"
            
            # Compare original and recovered files
            checksums_match = False
            if recovery_success and os.path.exists(recovered_path):
                original_md5 = self.calculate_md5(self.test_zip_path)
                recovered_md5 = self.calculate_md5(recovered_path)
                checksums_match = original_md5 == recovered_md5
                
                print(f"Recovery {'successful' if checksums_match else 'failed - checksums do not match'}")
            else:
                print(f"Recovery failed - file not found or decode error")
            
            # Calculate ratios
            ratio_before_youtube = file_info['output_size'] / file_info['input_size']
            ratio_after_youtube = youtube_size / file_info['input_size']
            
            # Prepare result
            result = {
                'test_id': test_id,
                'encoding_mode': file_info['encoding_mode'],
                'block_size': file_info['block_size'],
                'grid_w': file_info['grid_w'],
                'grid_h': file_info['grid_h'],
                'fps': file_info['fps'],
                'crf': file_info['crf'],
                'border': file_info['border'],
                'input_size_kb': file_info['input_size'] / 1024,
                'output_size_kb': file_info['output_size'] / 1024,
                'youtube_size_kb': youtube_size / 1024,
                'ratio_before_youtube': ratio_before_youtube,
                'ratio_after_youtube': ratio_after_youtube,
                'recovery_success': recovery_success,
                'checksums_match': checksums_match,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add to results dataframe and list
            self.results_df = pd.concat([self.results_df, pd.DataFrame([result])], ignore_index=True)
            results.append(result)
            
            print(f"Test completed - Ratio after YouTube: {ratio_after_youtube:.2f}x, "
                  f"Recovery success: {recovery_success and checksums_match}")
        
        # Save updated results
        self.results_df.to_csv(self.results_file, index=False)
        
        return results
    
    def run_test_case(self, encoding_mode, block_size, grid_w=None, grid_h=None, 
                     fps=60, crf=18, border=0, callback=None):
        """
        Run a single test case with specified parameters
        
        Args:
            encoding_mode: "bw", "4", or "16" for the encoding mode
            block_size: Size of each block in pixels
            grid_w, grid_h: Grid dimensions (calculated if None)
            fps: Video frame rate
            crf: Constant Rate Factor for video encoding (quality)
            border: Border width in pixels
            callback: Optional progress callback function
        
        Returns:
            Dictionary with test results
        """
        # Generate a unique test ID
        test_id = f"{encoding_mode}_{block_size}_{fps}_{crf}_{border}_{int(time.time())}"
        
        # Calculate grid dimensions if not provided
        if grid_w is None:
            grid_w = 3840 // block_size
        if grid_h is None:
            grid_h = 2160 // block_size
        
        # For manual mode, just encode the video and return placeholder
        if self.manual_mode:
            encode_result = self.encode_test_video(
                test_id, encoding_mode, block_size, grid_w, grid_h, 
                fps, crf, border, callback
            )
            # Return a placeholder result (processing will happen later in batch)
            return {
                'test_id': test_id,
                'encoding_mode': encoding_mode,
                'block_size': block_size,
                'grid_w': grid_w,
                'grid_h': grid_h,
                'fps': fps,
                'crf': crf,
                'border': border,
                'status': 'pending'
            }
        
        # For non-manual mode, continue with normal process
        video_info = self.encode_test_video(
            test_id, encoding_mode, block_size, grid_w, grid_h, 
            fps, crf, border, callback
        )
        
        if not video_info:
            return None
        
        # Step 3: Decode YouTube-compressed video back to ZIP
        recovered_path = self.recovered_dir / f"{test_id}_recovered.zip"
        print(f"Decoding compressed video back to ZIP...")
        decode_result = decode_video_to_zip(
            str(video_info['youtube_path']),
            str(recovered_path),
            grid_w,
            grid_h,
            border,
            block_size,
            callback=callback,
            encoding_mode=encoding_mode if encoding_mode in ["bw", "4", "16"] else encoding_mode
        )
        
        recovery_success = decode_result["status"] == "success"
        
        # Step 4: Compare original and recovered files
        checksums_match = False
        if recovery_success and os.path.exists(recovered_path):
            original_md5 = self.calculate_md5(self.test_zip_path)
            recovered_md5 = self.calculate_md5(recovered_path)
            checksums_match = original_md5 == recovered_md5
            
            print(f"Recovery {'successful' if checksums_match else 'failed - checksums do not match'}")
        else:
            print(f"Recovery failed - file not found or decode error")
        
        # Calculate ratios
        ratio_before_youtube = video_info['output_size'] / video_info['input_size']
        ratio_after_youtube = video_info['youtube_size'] / video_info['input_size']
        
        # Prepare results
        result = {
            'test_id': test_id,
            'encoding_mode': encoding_mode,
            'block_size': block_size,
            'grid_w': grid_w,
            'grid_h': grid_h,
            'fps': fps,
            'crf': crf,
            'border': border,
            'input_size_kb': video_info['input_size'] / 1024,
            'output_size_kb': video_info['output_size'] / 1024,
            'youtube_size_kb': video_info['youtube_size'] / 1024,
            'ratio_before_youtube': ratio_before_youtube,
            'ratio_after_youtube': ratio_after_youtube,
            'recovery_success': recovery_success,
            'checksums_match': checksums_match,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add to results dataframe
        self.results_df = pd.concat([self.results_df, pd.DataFrame([result])], ignore_index=True)
        self.results_df.to_csv(self.results_file, index=False)
        
        print(f"Test completed - Ratio after YouTube: {ratio_after_youtube:.2f}x, "
              f"Recovery success: {recovery_success and checksums_match}")
        
        return result
        
    def run_parameter_sweep(self, test_sets):
        """
        Run a parameter sweep across multiple settings
        
        Args:
            test_sets: List of dictionaries with parameters to test
        """
        results = []
        total_tests = len(test_sets)
        
        print(f"Starting parameter sweep with {total_tests} test configurations...")
        
        # Clear batch files list for manual mode
        if self.manual_mode:
            self.batch_files = []
        
        # First phase: encode all videos
        for i, params in enumerate(test_sets):
            print(f"\nTest {i+1}/{total_tests}")
            result = self.run_test_case(**params)
            if result:
                results.append(result)
        
        # For manual mode, wait for user to upload/download all videos
        if self.manual_mode and self.batch_files:
            print(f"\n{'*'*80}")
            print(f"MANUAL UPLOAD MODE: {len(self.batch_files)} videos have been encoded.")
            print(f"Please upload ALL videos to YouTube and download them.")
            print(f"\nFor each test video, please:")
            print(f"1. Upload the video to YouTube")
            print(f"2. Download the processed video")
            print(f"3. Save it with the same filename in the downloads directory")
            print(f"\nVideos to process:")
            
            # Create a file with upload instructions
            instructions_file = self.output_dir / "upload_instructions.txt"
            with open(instructions_file, 'w') as f:
                f.write("UPLOAD INSTRUCTIONS\n")
                f.write("==================\n\n")
                f.write("Please upload each of these videos to YouTube, then download them and save\n")
                f.write("with the specified filename in the manual_downloads directory.\n\n")
                
                for i, file_info in enumerate(self.batch_files):
                    source_file = file_info['video_path']
                    target_file = file_info['manual_download_path']
                    f.write(f"{i+1}. Upload: {source_file}\n")
                    f.write(f"   Save as: {target_file}\n\n")
                    
                    # Also print to console
                    print(f"{i+1}. Upload: {source_file}")
                    print(f"   Save downloaded as: {target_file}")
            
            print(f"\nDetailed instructions have been saved to: {instructions_file}")
            print(f"{'*'*80}")
            
            # Wait for user to complete uploads/downloads
            while True:
                user_input = input("\nHave you completed uploading and downloading all videos? (y/n): ").strip().lower()
                if user_input == 'y':
                    # Check if at least some files exist
                    found_files = 0
                    for file_info in self.batch_files:
                        if os.path.exists(file_info['manual_download_path']):
                            found_files += 1
                    
                    if found_files == 0:
                        print("Warning: No downloaded files found. Are you sure you saved them in the correct location?")
                        continue
                    elif found_files < len(self.batch_files):
                        proceed = input(f"Only {found_files}/{len(self.batch_files)} files found. Proceed anyway? (y/n): ").strip().lower()
                        if proceed != 'y':
                            continue
                    
                    print(f"Found {found_files}/{len(self.batch_files)} downloaded files. Proceeding with analysis...")
                    break
                elif user_input == 'n':
                    print("Please complete the upload/download process before continuing.")
                    continue
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
                    continue
            
            # Second phase for manual mode: process all downloaded YouTube videos
            process_results = self.process_youtube_videos()
            
            # Replace the placeholder results with actual results
            results = process_results
        
        # Find the best settings (lowest ratio with successful recovery)
        successful_results = [r for r in results if r.get('recovery_success', False) and r.get('checksums_match', False)]
        
        if successful_results:
            # Sort by ratio after YouTube compression
            best_results = sorted(successful_results, key=lambda x: x.get('ratio_after_youtube', float('inf')))
            
            # Print top 5 best results
            print("\n=== TOP 5 BEST CONFIGURATIONS ===")
            for i, result in enumerate(best_results[:5]):
                print(f"{i+1}. Mode: {result['encoding_mode']}, Block Size: {result['block_size']}, "
                      f"Grid: {result['grid_w']}x{result['grid_h']}, FPS: {result['fps']}, "
                      f"CRF: {result['crf']}, Border: {result['border']}")
                if 'ratio_after_youtube' in result:
                    print(f"   Ratio: {result['ratio_after_youtube']:.2f}x, "
                          f"Original Size: {result['input_size_kb']:.1f}KB, "
                          f"Final Size: {result['youtube_size_kb']:.1f}KB")
            
            # Save summary of best results
            with open(self.summary_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'total_tests': total_tests,
                    'successful_tests': len(successful_results),
                    'best_configurations': [
                        {k: v for k, v in r.items() if k != 'status'} 
                        for r in best_results[:10]
                    ]
                }, f, indent=2)
            
            return best_results
        else:
            print("No successful recoveries found!")
            return []
    
    def visualize_results(self):
        """Generate visualization plots of test results"""
        if len(self.results_df) == 0:
            print("No results to visualize!")
            return
        
        # Filter only successful tests
        success_df = self.results_df[self.results_df['recovery_success'] & self.results_df['checksums_match']]
        
        if len(success_df) == 0:
            print("No successful tests to visualize!")
            return
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Ratio by Encoding Mode
        plt.figure(figsize=(12, 6))
        modes = success_df['encoding_mode'].unique()
        
        data = []
        labels = []
        for mode in modes:
            mode_data = success_df[success_df['encoding_mode'] == mode]['ratio_after_youtube']
            if len(mode_data) > 0:
                data.append(mode_data)
                labels.append(f"{mode} (n={len(mode_data)})")
        
        if data:
            plt.boxplot(data, labels=labels)
            plt.title('Compression Ratio by Encoding Mode')
            plt.ylabel('Ratio (Output/Input)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(plots_dir / "ratio_by_mode.png", dpi=300)
        
        # 2. Ratio by Block Size
        plt.figure(figsize=(12, 6))
        block_sizes = sorted(success_df['block_size'].unique())
        
        data = []
        labels = []
        for bs in block_sizes:
            bs_data = success_df[success_df['block_size'] == bs]['ratio_after_youtube']
            if len(bs_data) > 0:
                data.append(bs_data)
                labels.append(f"{bs}px (n={len(bs_data)})")
        
        if data:
            plt.boxplot(data, labels=labels)
            plt.title('Compression Ratio by Block Size')
            plt.ylabel('Ratio (Output/Input)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(plots_dir / "ratio_by_block_size.png", dpi=300)
        
        # 3. Success Rate by Mode and Block Size
        plt.figure(figsize=(12, 6))
        
        mode_bs_counts = {}
        for mode in modes:
            mode_bs_counts[mode] = {}
            for bs in block_sizes:
                # Calculate success rate for this combination
                total = len(self.results_df[(self.results_df['encoding_mode'] == mode) & 
                                           (self.results_df['block_size'] == bs)])
                success = len(success_df[(success_df['encoding_mode'] == mode) & 
                                        (success_df['block_size'] == bs)])
                
                if total > 0:
                    mode_bs_counts[mode][bs] = success / total * 100
        
        # Plot
        for mode in modes:
            if mode_bs_counts[mode]:
                x = list(mode_bs_counts[mode].keys())
                y = list(mode_bs_counts[mode].values())
                plt.plot(x, y, 'o-', label=mode)
        
        plt.xlabel('Block Size')
        plt.ylabel('Success Rate (%)')
        plt.title('Recovery Success Rate by Mode and Block Size')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(plots_dir / "success_rate.png", dpi=300)
        
        # 4. Scatter plot: CRF vs Ratio
        plt.figure(figsize=(12, 6))
        
        for mode in modes:
            mode_df = success_df[success_df['encoding_mode'] == mode]
            plt.scatter(mode_df['crf'], mode_df['ratio_after_youtube'], 
                       label=mode, alpha=0.7)
        
        plt.xlabel('CRF Value')
        plt.ylabel('Ratio After YouTube (Output/Input)')
        plt.title('CRF vs Compression Ratio')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(plots_dir / "crf_vs_ratio.png", dpi=300)
        
        print(f"Visualizations saved to {plots_dir}")


def progress_callback(progress, message):
    """Simple progress callback function"""
    if progress < 0:
        print(f"\rError: {message}")
    elif progress == 100:
        print(f"\rCompleted: {message}")
    else:
        print(f"\r{message} - {progress:.1f}%", end="")


def main():
    """Main function to run the tests"""
    import argparse
    parser = argparse.ArgumentParser(description="Test Zip2Vid YouTube compression settings")
    parser.add_argument("--zip", "-z", required=True, help="Path to test ZIP file")
    parser.add_argument("--output", "-o", default="test_results", help="Output directory")
    parser.add_argument("--quick", "-q", action="store_true", help="Run quick test with fewer parameters")
    parser.add_argument("--visualize", "-v", action="store_true", help="Only visualize existing results")
    parser.add_argument("--manual", "-m", action="store_true", help="Manual YouTube upload mode")
    
    # Fixed parameter options
    parser.add_argument("--fps", type=int, help="Fix FPS to this value")
    parser.add_argument("--block-size", type=int, help="Fix block size to this value")
    parser.add_argument("--crf", type=int, help="Fix CRF value to this value")
    parser.add_argument("--border", type=int, help="Fix border value to this value")
    parser.add_argument("--encoding-mode", choices=["bw", "4", "16"], help="Fix encoding mode")
    parser.add_argument("--grid-w", type=int, help="Fix grid width")
    parser.add_argument("--grid-h", type=int, help="Fix grid height")
    
    args = parser.parse_args()
    
    tester = YouTubeCompressionTester(args.zip, args.output, manual_mode=args.manual)
    
    if args.visualize:
        tester.visualize_results()
        return
    
    # Build fixed parameters dict from command line args
    fixed_params = {}
    if args.fps is not None:
        fixed_params["fps"] = args.fps
    if args.block_size is not None:
        fixed_params["block_size"] = args.block_size
    if args.crf is not None:
        fixed_params["crf"] = args.crf
    if args.border is not None:
        fixed_params["border"] = args.border
    if args.encoding_mode is not None:
        fixed_params["encoding_mode"] = args.encoding_mode
    if args.grid_w is not None:
        fixed_params["grid_w"] = args.grid_w
    if args.grid_h is not None:
        fixed_params["grid_h"] = args.grid_h
    
    # Define test parameters
    if args.quick:
        # Quick test with fewer parameters
        if fixed_params:
            # With fixed parameters, adjust the quick test to respect them
            variable_params = {
                "encoding_mode": ["bw", "4", "16"] if "encoding_mode" not in fixed_params else [fixed_params["encoding_mode"]],
                "block_size": [2, 4, 8, 16, 24] if "block_size" not in fixed_params else [fixed_params["block_size"]],
                "fps": [60] if "fps" not in fixed_params else [fixed_params["fps"]],
                "crf": [12, 18, 24] if "crf" not in fixed_params else [fixed_params["crf"]],
                "border": [0] if "border" not in fixed_params else [fixed_params["border"]]
            }
            
            best_results = tester.run_parameter_sweep(fixed_params=fixed_params, variable_params=variable_params)
        else:
            # Traditional quick test
            test_sets = [
                # Black & White tests
                #{"encoding_mode": "bw", "block_size": 1, "fps": 60, "crf": 16, "border": 0},
                {"encoding_mode": "bw", "block_size": 1, "fps": 60, "crf": 18, "border": 0},
                #{"encoding_mode": "bw", "block_size": 2, "fps": 60, "crf": 24, "border": 0},
                #{"encoding_mode": "bw", "block_size": 2, "fps": 60, "crf": 28, "border": 0},
                #{"encoding_mode": "bw", "block_size": 2, "fps": 60, "crf": 16, "border": 0},
                {"encoding_mode": "bw", "block_size": 2, "fps": 60, "crf": 18, "border": 0},
                #{"encoding_mode": "bw", "block_size": 4, "fps": 60, "crf": 18, "border": 0},
                #{"encoding_mode": "bw", "block_size": 4, "fps": 60, "crf": 21, "border": 0},
                {"encoding_mode": "bw", "block_size": 4, "fps": 60, "crf": 24, "border": 0},
                #{"encoding_mode": "bw", "block_size": 8, "fps": 60, "crf": 18, "border": 0},
                #{"encoding_mode": "bw", "block_size": 8, "fps": 60, "crf": 21, "border": 0},
                #{"encoding_mode": "bw", "block_size": 8, "fps": 60, "crf": 24, "border": 0},
                
                # 4-Gray tests
                {"encoding_mode": "4", "block_size": 4, "fps": 60, "crf": 18, "border": 0},
                #{"encoding_mode": "4", "block_size": 8, "fps": 60, "crf": 18, "border": 0},
                {"encoding_mode": "4", "block_size": 8, "fps": 60, "crf": 24, "border": 0},
                #{"encoding_mode": "4", "block_size": 12, "fps": 60, "crf": 18, "border": 0},
                #{"encoding_mode": "4", "block_size": 12, "fps": 60, "crf": 21, "border": 0},
                
                # 16-Color tests
                #{"encoding_mode": "16", "block_size": 16, "fps": 60, "crf": 18, "border": 0},
                {"encoding_mode": "16", "block_size": 20, "fps": 60, "crf": 18, "border": 0},
                {"encoding_mode": "16", "block_size": 20, "fps": 60, "crf": 18, "border": 1},
                #{"encoding_mode": "16", "block_size": 20, "fps": 60, "crf": 21, "border": 0},
                #{"encoding_mode": "16", "block_size": 20, "fps": 60, "crf": 24, "border": 0},
                {"encoding_mode": "16", "block_size": 24, "fps": 60, "crf": 18, "border": 0},
            ]
            best_results = tester.run_parameter_sweep(test_sets)
    else:
        # Comprehensive parameter sweep
        if fixed_params:
            # With fixed parameters
            variable_params = {}
            
            # Only include variable parameters that are not fixed
            if "encoding_mode" not in fixed_params:
                variable_params["encoding_mode"] = ["bw", "4", "16"]
            if "block_size" not in fixed_params:
                variable_params["block_size"] = [2, 4, 6, 8, 10, 12, 16, 20, 24]
            if "fps" not in fixed_params:
                variable_params["fps"] = [30, 60]
            if "crf" not in fixed_params:
                variable_params["crf"] = [16, 18, 21, 24]
            if "border" not in fixed_params:
                variable_params["border"] = [0, 1]
            
            # If everything is fixed, add at least one variable parameter
            if not variable_params:
                print("Warning: All parameters are fixed. Adding CRF as variable parameter.")
                variable_params["crf"] = [16, 18, 21, 24]
            
            # Generate all combinations of variable parameters
            test_sets = []
            # Logic to generate test sets from variable_params
            # This would need to be implemented based on how your existing parameter sweep works
            
            best_results = tester.run_parameter_sweep(test_sets)
        else:
            # Traditional comprehensive parameter sweep
            test_sets = []
            
            # Black & White mode tests
            for block_size in [2, 4, 6, 8, 10, 12, 16]:
                for crf in [16, 18, 21, 24]:
                    for fps in [30, 60]:
                        for border in [0, 1]:
                            test_sets.append({
                                "encoding_mode": "bw",
                                "block_size": block_size,
                                "fps": fps,
                                "crf": crf,
                                "border": border
                            })
            
            # 4-Gray mode tests
            for block_size in [4, 8, 12, 16]:
                for crf in [16, 18, 21, 24]:
                    for fps in [30, 60]:
                        for border in [0, 1]:
                            test_sets.append({
                                "encoding_mode": "4",
                                "block_size": block_size,
                                "fps": fps,
                                "crf": crf,
                                "border": border
                            })
            
            # 16-Color mode tests
            for block_size in [8, 12, 16, 20, 24]:
                for crf in [16, 18, 21, 24]:
                    for fps in [30, 60]:
                        for border in [0, 1]:
                            test_sets.append({
                                "encoding_mode": "16",
                                "block_size": block_size,
                                "fps": fps,
                                "crf": crf,
                                "border": border
                            })
                    
            best_results = tester.run_parameter_sweep(test_sets)
    
    # Visualize results
    tester.visualize_results()
    
    return best_results


if __name__ == "__main__":
    main()