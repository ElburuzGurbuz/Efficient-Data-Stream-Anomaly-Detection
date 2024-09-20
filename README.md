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
