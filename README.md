# Augmented-reality-simulation-of-cardiovascular-interventional-surgery-IPP-
Augmented reality simulation of cardiovascular interventional surgery Mar 2020- May 2021


- Implemented the framework of surgery training assistant system based on augmented reality.
- Predicted the operation trajectory using LSTM and use kdtree to calculate the distance for operation safety warning.(This part is not shown in this code repo for some privacy reason.)
- Displayed the vascular model in augmented reality with OpenGL, and design UI interface to adjust the size, rotation and translation of the model.
- The marker of aruco Library in OpenCV is used to realize the coordinate positioning of QR code.


## How to run the code

It's recommend to use visual studio environment, and the main function is in `./TestVision/main.cpp`. 
Also makefile is provided in the`./TestVision` directory. 

Some of the 3d vessel model file is in the `./model` directory.


## Source
It's part of the work in this project. [Vascular Intervention Training System Based on Electromagnetic Tracking Technology, 2020 International Conference on Virtual Reality and Visualization (ICVRV)](
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9479727)
