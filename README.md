# PyRay

![Alt text](assets/example.png?raw=true "Title")

A ray tracing engine for python

# Features
- Metal Acceleration
- Spheres and Triangles
- Diffuse and Specular Reflections
- Basic HDR Tone Mapping
- Realtime Mode
- Save Higher Quality Renders

# Getting Started
You must have Python installed (tested with Python 3.10).
1. Clone the repo
```
git clone https://github.com/igm503/PyRay.git
```
2. Install the required libraries
```
cd PyRay && pip install -r requirements.txt

# if you have a metal gpu
pip install pyobjc

# if you have an nvidia gpu, install the appropriate cupy library for your system. e.g.
pip install cupy-cuda11x
```
3. Run the example scene
```
cd PyRay && python demo.py default.yaml --device [cpu|metal|cuda]
```

# Interacting with the Scene
When running the renderer, you can move the camera around and take other actions through the keyboard.

## Navigation
Use `W`, `A`, `S`, `D` to move the camera position, and `I`, `J`, `K`, `L` to adjust the camera's viewing direction.

## Rendering
Press `R` to render the scene at high quality using the settings specified in the configuration file. This will save the rendered image (as well as intermediate renders) in directory specified in the scene config. This may take a while, depending on the parameters for saved renders you've set in your [configuration file](#configuring-the-scene), and you won't be able to control the camera until the render is completed. There will be a progress bar in your console.

## All Controls

| Key | Function |
|-----|----------|
| `Q` | Quit the application |
| `W` | Move camera forward |
| `S` | Move camera backward |
| `A` | Move camera left |
| `D` | Move camera right |
| `<space>` | Move camera up |
| `<z>` | Move camera down |
| `I` | Look up |
| `K` | Look down |
| `J` | Look left |
| `L` | Look right |
| `R` | Save a high quality render of the current view |
| `=` | Decrease field of view (zoom in) |
| `-` | Increase field of view (zoom out) |
| `` ` `` | Toggle debug mode |

# Configuring the Scene
You can create a custom scene by creating a yaml file in ```scenes/``` and configuring it as follows
```yaml

view:                                # parameters for virtual camera

  origin: [-30, -14.0, 5.0]          # initial cam position
  dir: [0.1, 0.1, 0]                 # initial cam direction
  fov: 70                            # field of view (degrees) of rendering
  exposure: 3.0                      # brightness of render
  num_samples: 10                    # number of rays to simulate per pixel
  max_bounces: 50                    # number of bounces each ray can travel
  render_resolution: [1280, 720]     # resolution of render
  display_resolution: [2560, 1440]   # resolution to display render at

save_render:                         # parameters for high quality saved renders
  save_path: renders/default         # directory to save in
  width: 3840                  
  height: 2160
  exposure: 3.0
  num_samples: 1000
  max_bounces: 100
  resolution: [1280, 720]            # resolution of iterative render

surfaces:                            # list of objects in scene

  - type: triangle                   # can be "triangle" or "sphere"
    points:                          # vertices of triangle
      - [-15000.0, -15000.0, 0]
      - [15000, 15000.0, -0]
      - [-15000.0, 15000, -0]
    material:                        
      luminance: 0.0                 # how much light the object emits (default 0)
      color: [0, 0.5, 0]             # RGB with range [0, 1]
      reflectivity: 0.0              # float [0, 1] with 0 being diffuse and 1 mirror-like (default 0)
      # TRANSPARENCY
      transparent: false             # bool whether the object is transparent
      refractive_index: 1.0          # index of refraction (default 1.0; only relevant for transparent objects)
      translucency: 0.00             # 0.0 is clear; higher values (> 0.02) becoming very foggy (only relevant for transparent objects)
      absorption: [0.0, 0.0, 0.0]    # exponential absorption rate of RBG light as it passes through material (only relevant for transparent objects)
      # GLOSSINESS
      glossy: false                   # bool whether the object is glossy (only relevant for non-transparent objects)
      gloss_refractive_index: 1.0    # index of refraction (default 1.0; only relevant for glossy objects)
      gloss_translucency: 0.00       # float [0, 1] controls gloss reflections, similar to reflectivity (only relevant for glossy objects)


  - type: sphere
    center: [700.0, 500.0, -200]     # center point of sphere
    radius: 500
    material:                        # material properties options are the same for spheres and triangles
      color: [0.7, 0, 0.5]
      transparency: 0.9
      refractive_index: 1.1
      translucency: 0.00

    ...                              

  - type: sphere
    center: [700.0, 500.0, 200]    
    radius: 500
    material:
      color: [0.7, 0, 0.5]
      reflectivity: 1.0

```
Render your scene with
```
python demo.py scene_name.yaml
```
By default, the renderer will use your cpu. If you have a GPU with metal support, you can use it for faster rendering like so:
```
python demo.py scene_name.yaml --device metal
```
If you have an Nvidia GPU, you can use it for faster rendering like so:
```
python demo.py scene_name.yaml --device cuda
```


# TODO
- Some Sort of Support for external scene formats
- Transparency
