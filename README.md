# PyRay
ray tracing engine for python

![Alt text](assets/example.png?raw=true "Title")

# WIP
- CUDA Acceleration [completed for metal devices; todo: cuda]
- Some Sort of Mesh Support
- More Materials (e.g. specular-diffuse combo)
- More Accurate Luminosity Handling

# Completed
- Support for Spheres and Triangles
- Metal Acceleration
- Diffuse and Specular Reflections
- Basic HDR Tone Mapping

# Installation
You must have Python installed (tested with Python 3.10).
1. Clone the repo
```git clone https://github.com/igm503/PyRay.git```
2. Install the requiremed libraries
```cd PyRay && pip install -r requirements.txt```
3. Run the example scene
```cd PyRay && python main.py```
