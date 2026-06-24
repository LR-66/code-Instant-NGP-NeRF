"""Real-time rendering efficiency evaluation."""

import time
import torch
import numpy as np


def compute_fps(render_func, num_frames: int = 100) -> dict:
    """
    Compute average FPS and standard deviation.
    
    Args:
        render_func: Callable that renders a frame
        num_frames: Number of frames to render
        
    Returns:
        dict with avg_fps, std_fps, frame_times
    """
    frame_times = []
    
    # Warm-up
    for _ in range(10):
        render_func()
    
    # Measure
    for _ in range(num_frames):
        start = time.perf_counter()
        render_func()
        end = time.perf_counter()
        frame_times.append(end - start)
    
    fps_values = [1.0 / t for t in frame_times]
    
    return {
        'avg_fps': np.mean(fps_values),
        'std_fps': np.std(fps_values),
        'min_fps': np.min(fps_values),
        'max_fps': np.max(fps_values),
        'frame_times': frame_times,
    }


def compute_m2p_latency(interact_func, num_trials: int = 100) -> dict:
    """
    Compute M2P (Motion-to-Photon) latency.
    
    Args:
        interact_func: Callable that processes one interaction
        num_trials: Number of trials
        
    Returns:
        dict with avg_latency, std_latency, latencies
    """
    latencies = []
    
    for _ in range(num_trials):
        start = time.perf_counter()
        interact_func()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        'avg_latency': np.mean(latencies),
        'std_latency': np.std(latencies),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies),
        'latencies': latencies,
    }