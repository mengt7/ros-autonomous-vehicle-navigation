# ROS Autonomous Vehicle Navigation 

This repository contains a ROS-based autonomous navigation system for an Ackermann-steered vehicle. The project implements real-time navigation, obstacle avoidance, and occupancy grid mapping using LiDAR and optional depth-camera data.

The system was developed and evaluated using both simulation and on-hardware testing, with an emphasis on algorithm correctness, stability, and real-time performance.

---

## Overview

The navigation pipeline processes LiDAR and sensor data to generate steering and speed commands for autonomous driving in structured environments such as hallways and indoor tracks. A simulation-first approach was used to develop and validate navigation behavior prior to deployment on embedded hardware.

---

## Key Capabilities

- Real-time LiDAR-based navigation and obstacle avoidance
- Gap-following and virtual-barrier navigation strategies
- Occupancy grid mapping using inverse sensor models (log-odds)
- Simulation-based testing using the F1TENTH simulator
- Visualization and debugging with RViz
- Deployment and testing on NVIDIA Jetson Nano hardware

---

## Development & Validation Workflow

The project follows an iterative development workflow:

1. **Simulation**
   - Navigation algorithms are developed and tested in the F1TENTH simulator
   - Vehicle behavior and sensor feedback are observed in a virtual environment

2. **Data Logging & Analysis**
   - Telemetry such as speed, steering angle, and virtual wall distances is logged
   - Logged data is used to assess stability, responsiveness, and correctness

3. **On-Hardware Testing**
   - Algorithms are deployed to embedded hardware
   - Additional debugging focuses on sensor synchronization, timing, and real-world constraints

This approach enables safer testing and faster iteration before hardware deployment.

---

## Tech Stack

- **ROS** (Python, C++)
- **F1TENTH simulator**
- **LiDAR**, RGB-D camera
- **RViz**
- **NVIDIA Jetson Nano**

---

## Project Structure

```
ros-autonomous-vehicle-navigation/
├─ wall_following.py          # LiDAR wall‑following node
├─ navigation_vb.py           # Gap + virtual‑barrier navigation
├─ navigation_vb_cam.py       # Gap‑barrier navigation with depth camera
├─ occupancygridmap.py        # Occupancy grid mapping (log-odds)
├─ params.yaml                # Tunable parameters & topic names
└─ README.md                  # 
```

---

## Notes

- This repository represents an academic robotics project.
- Code is organized for experimentation, clarity, and validation rather than production deployment.
