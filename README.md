# edge-detection
Python project for edge detection using classical filters such as Sobel, Prewitt, Roberts and Laplacian. It also includes thresholding methods and gradient visualization to understand contour extraction.


## Features
- Sobel filter
- Prewitt filter
- Roberts filter
- Laplacian filter
- Laplacian of Gaussian (LoG)
- Thresholding (simple and hysteresis)
- Gradient visualization (Gx, Gy, magnitude, phase)
## Examples

### Sobel Filter
![Sobel](Screenshot (1131).png)

The Sobel filter detects edges by computing image gradients in both horizontal and vertical directions.

### Result after thresholding
![Result](Screenshot (1132).png)

After applying a threshold, weak edges are removed and only strong contours remain.

### Gradient Visualization
![Gradient](Screenshot (1133).png)

This shows Gx, Gy, magnitude and phase to understand how edges are computed.


### Additional result
![Extra](Screenshot (1134).png)
## How to run
pip install pillow numpy
python app.py

## Notes
This project was done as part of my multimedia studies to understand edge detection techniques.

## Author
Harrouche Bassma

