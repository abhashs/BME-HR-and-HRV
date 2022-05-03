Remote BPM and HRV Detection and Overlay
-----------------------

Extension of [webcam-pulse-detector](https://github.com/thearn/webcam-pulse-detector), provides an additional GUI overlay with virtual camera, as well as HRV estimations.


Requirements:
---------------

- [Python v2.7 or v3.5+)](http://python.org/)
- [OpenCV v2+](http://opencv.org/)
- Numpy, Scipy
- pyvirtualcam

Quickstart:
------------

- run `get_pulse.py` to start the application

```
python get_pulse.py

Usage notes:
----------
- When run, a window will open showing a stream from your computer's webcam
- When a forehead location has been isolated, the user should press "S" on their 
keyboard to lock this location, and remain as still as possible (the camera 
stream window must have focus for the click to register). This freezes the acquisition location in place. This lock can
be released by pressing "S" again.
- To view a stream of the measured data as it is gathered, press "D". To hide this display, press "D" again.
- The data display shows three data traces, from top to bottom: 
   1. raw optical intensity
   2. extracted heartbeat signal
   3. Power spectral density, with local maxima indicating the heartrate (in beats per minute). 
- With consistent lighting and minimal head motion, a stable heartbeat should be 
isolated in about 15 to 20 seconds. A count-down is shown in the image frame.
- If a large spike in optical intensity is measured in the data (due to motion 
noise, sudden change in lighting, etc) the data collection process is reset and 
started over. The sensitivity of this feature can be tweaked by changing `data_spike_limit` on line 31 of [get_pulse.py](get_pulse.py).
Other mutable parameters of the analysis can be changed here as well.
