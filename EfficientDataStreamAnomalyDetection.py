"""
Efficient Data Stream Anomaly Detection

This script implements real-time anomaly detection using the Exponential Weighted
Moving Average (EWMA) algorithm with dynamic thresholding.

The Exponential Weighted Moving Average (EWMA) algorithm was chosen for this anomaly 
detection task for several key reasons:

1. Adaptability: Adjusts to gradual changes in data distribution.
2. Efficiency: Minimal memory and processing requirements for real-time processing.
3. Sensitivity Control: Adjustable smoothing factor (alpha) for fine-tuning.
4. Robust to Noise: Smooths short-term fluctuations while tracking long-term trends.
5. Simple Implementation: Easy to understand and maintain.
6. Proven Track Record: Widely used in finance, manufacturing, and system monitoring.

By combining EWMA with dynamic thresholding, this implementation can effectively 
detect both sudden spikes and gradual anomalies in real-time, making it well-suited 
for the continuous data stream anomaly detection task.

Other algorithms were not chosen for the following reasons:

1. Static Threshold Methods: Unlike EWMA, static thresholds can't adapt to changing data 
   distributions, leading to increased false positives or negatives over time.

2. Machine Learning Models (e.g., Isolation Forests, One-Class SVM): While powerful, 
   these require significant training data and computational resources, making them 
   less suitable for real-time processing of continuous data streams.

3. Sliding Window Techniques: These methods, while adaptive, often struggle with 
   gradual anomalies and require careful window size selection, which can be challenging 
   in dynamic environments.

4. Statistical Methods (e.g., Z-score): Simple statistical methods often assume a 
   normal distribution, which may not hold for all data streams, potentially leading 
   to inaccurate anomaly detection.

Example:
Consider a temperature sensor in a manufacturing process. The normal operating 
temperature fluctuates between 20°C and 25°C, with occasional spikes up to 30°C 
during peak operations. A gradual increase to 35°C over several hours might indicate 
a serious issue.

- A static threshold at 30°C would miss the gradual increase to 35°C.
- A simple statistical method like Z-score might flag normal spikes to 30°C as anomalies.
- A sliding window technique might miss the gradual increase if the window is too small.

EWMA, however, can adapt to the normal fluctuations while still detecting both sudden 
spikes and gradual increases, making it ideal for this scenario.

"""

import random
import math
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class EWMADetector:
    
    """
    A class for detecting anomalies in a continuous data stream using EWMA.

    This implementation uses a sliding window approach combined with EWMA
    for adaptive anomaly detection.
    """

    def __init__(self, window_size=30, smoothing_factor=0.2, threshold_multiplier=2):
        
        """
        Initialize the EWMADetector.

        Args:
            window_size (int): Size of the sliding window for calculations.
            smoothing_factor (float): The weighting factor for EWMA (0 < smoothing_factor <= 1).
            threshold_multiplier (float): Multiplier for the standard deviation to set thresholds.
        
        """
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self.threshold_multiplier = threshold_multiplier
        self.data_window = deque(maxlen=window_size)
        self.ewma = None
        self.mse = None

    def update(self, value):
        
        """
        Update the detector with a new data point and check for anomalies.

        Args:
            value (float): The new data point.

        Returns:
            bool: True if an anomaly is detected, False otherwise.
        """
        
        self.data_window.append(value)

        # Wait until we have enough data points
        if len(self.data_window) < self.window_size:
            return False

        # Initialize EWMA and MSE if this is the first full window
        if self.ewma is None:
            self.ewma = sum(self.data_window) / self.window_size
            self.mse = sum((x - self.ewma) ** 2 for x in self.data_window) / self.window_size
        else:
            # Update EWMA and MSE
            self.ewma = (self.smoothing_factor * value) + ((1 - self.smoothing_factor) * self.ewma)
            self.mse = (self.smoothing_factor * (value - self.ewma) ** 2) + ((1 - self.smoothing_factor) * self.mse)

        # Calculate threshold and check for anomaly
        threshold = self.threshold_multiplier * math.sqrt(self.mse)
        is_anomaly = abs(value - self.ewma) > threshold

        # Debugging print to check when an anomaly is detected
        if is_anomaly:
            print(f"Anomaly detected: Value={value}, EWMA={self.ewma}, Threshold={threshold}")

        return is_anomaly

def generate_data_stream():
   
    """
    Generate a simulated data stream with occasional anomalies.

    Yields:
        float: A data point from the simulated stream.
    """
    base = 100
    while True:
        # Normal fluctuation
        value = base + random.gauss(0, 10)

        # Introduce occasional anomalies (5% chance)
        if random.random() < 0.05:
            value += random.choice([-1, 1]) * random.uniform(50, 100)

        yield value

def main():
    
    """
    Main function to run the anomaly detection visualization.
    """
    
    detector = EWMADetector()
    data_stream = generate_data_stream()

    data = []
    anomalies = []

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], lw=2, color='black', label='Data')
    scatter = ax.scatter([], [], color='blue', s=100, zorder=5, label='Anomalies')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_title('Data Stream with Anomalies (EWMA-based Detection)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()

    def update(frame):
        
        """
        Update function for the animation.

        Args:
            frame: The current frame number (not used, but required by FuncAnimation).

        Returns:
            tuple: Updated line and scatter plot objects.
        """
        
        value = next(data_stream)
        data.append(value)

        if detector.update(value):
            anomalies.append(len(data) - 1)

        # Update the plot data
        x = list(range(max(0, len(data) - 200), len(data)))
        y = data[-200:]
        line.set_data(x, y)

        anomaly_x = [i for i in anomalies if i >= len(data) - 200]
        anomaly_y = [data[i] for i in anomaly_x]

        # Update scatter data correctly by converting to a NumPy array
        scatter.set_offsets(np.c_[anomaly_x, anomaly_y])

        # Adjust plot limits
        ax.set_xlim(max(0, len(data) - 200), len(data))
        ax.set_ylim(min(y) - 10, max(y) + 10)

        return line, scatter

    # Create and display the animation
    ani = FuncAnimation(fig, update, frames=None, blit=True, interval=50, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()
    
    
