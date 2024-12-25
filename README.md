# PyRay

![Alt text](assets/nesting.jpeg?raw=true "Title")

A ray tracing engine for python.

# Features
- Metal and Cuda Acceleration
- Spheres, Triangles, and Quads
- Meshes from Triangles and Quads
- Diffuse and Specular Reflections
- Transparency and Refraction
- Simple HDR Tone Mapping
- Realtime and Cumulative Render Modes

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
cd PyRay && python demo.py default.yaml --device [metal|cuda]
```
Note: there is a cpu engine, but it is not optimized and is extraordinarily slow (on an M2 max chip, running on the cpu is 10,000-100,000x slower than running on the gpu) and not recommended.

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
  background_color: [0.5, 0.8, 0.9]  # background color for rays that hit nothing
  backgorund_luminance: 1.0          # luminance of background for rays that hit nothing
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

  # SINGLE OBJECTS
  - type: triangle                   # can be "triangle", "sphere", or "quad"
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

  # MESHES -- allow you to define triangle or quad meshes that have a homogeneous internal material

  - type: mesh              
    surfaces:
      - type: quad                    # can be "triangle" or "quad"
        points:
          - [0.0, 0.0, 5.0]
          - [0.0, 5.0, 5.0]
          - [5.0, 5.0, 5.0]
          - [5.0, 0.0, 5.0]
      - type: quad
        points:
          - [0.0, 0.0, 8.0]
          - [0.0, 5.0, 8.0]
          - [5.0, 5.0, 8.0]
          - [5.0, 0.0, 8.0]
      - type: quad
        points:
          - [0.0, 0.0, 5.0]
          - [0.0, 0.0, 8.0]
          - [0.0, 5.0, 8.0]
          - [0.0, 5.0, 5.0]
      - type: quad
        points:
          - [5.0, 0.0, 5.0]
          - [5.0, 0.0, 8.0]
          - [5.0, 5.0, 8.0]
          - [5.0, 5.0, 5.0]
      - type: quad
        points:
          - [0.0, 0.0, 5.0]
          - [0.0, 0.0, 8.0]
          - [5.0, 0.0, 8.0]
          - [5.0, 0.0, 5.0]
      - type: quad
        points:
          - [0.0, 5.0, 5.0]
          - [0.0, 5.0, 8.0]
          - [5.0, 5.0, 8.0]
          - [5.0, 5.0, 5.0]
    material:                         # material properties for all surfaces in the mesh. Options are the same as for single surfaces.
      color: [1.0, 1.0, 1.0]
      reflectivity: 0.9
      transparent: true
      refractive_index: 1.125
      absorption: [0.6, 0.6, 0.2]
      translucency: 0.0

```
Render your scene with
```
python demo.py scene_name.yaml --device [metal|cuda]
```

# Material Properties

Surfaces can have the following properties:

```yaml
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
```

## Color
![Alt text](assets/color.jpeg?raw=true "color")

The `color` property is a `list[float] [0, 1]` that determines the RGB values of the object.
- `Color` affects diffuse and reflective interactions but does not affect light that is transmitted through objects or reflections due to gloss.
- Strictly speaking, `color` values can take values outside of [0, 1] but values between 0 and 1 are easier to reason about.

To run the pictured scene:
```
python demo.py color.yaml --device [metal|cuda]
```

## Luminance
![Alt text](assets/luminance.jpeg?raw=true "luminance")

The `luminance` property is a `float [0, inf]` that determines the amount of light emitted by the object. 
- Surfaces with `luminance` greater than 0 will terminate ray bounces and give their color and intensity to the light ray.
- Surfaces with `luminance` greater than 0 will not reflect any `gloss`, `reflectivity`, or `transparency` attributes.

To run the pictured scene:
```
python demo.py luminance.yaml --device [metal|cuda]
```

## Reflectivity
![Alt text](assets/reflectivity.jpeg?raw=true "reflectivity")

The `reflectivity` property is a `float [0, 1]` that determines how reflective a surface is.

- Surfaces with `reflectivity` equal to 0 will perfectly diffusely reflect incoming rays.
- Surfaces with `reflectivity` equal to 1 will perfectly specularly reflect incoming rays.
- Surfaces with `reflectivity` between 0 and 1 will mix diffuse and specular reflection.
- Reflective surfaces give their color to the reflected ray.

To run the pictured scene:
```
python demo.py reflectivity.yaml --device [metal|cuda]
```

## Transparency
![Alt text](assets/nesting.jpeg?raw=true "transparent")

Transparent surfaces allow light to pass through. `Transparency` is a `bool`, and transparent surfaces, provided they do not have a non-zero luminance value, ignore all other properties other than `absorption`, `refractive_index`, and `translucency`.

### Absorption
![Alt text](assets/absorption.jpeg?raw=true "absorption")

The `absorption` property is a `list[float] [0, 1]` that determines the amount of light of each color (RGB) that is absorbed by the object. A value of 1 will cause all light to be absorbed over any distance; if the value is 0, none of that color will be absorbed over any distance.

To run the pictured scene:
```
python demo.py absorption.yaml --device [metal|cuda]
```

### Refractive Index
![Alt text](assets/refractive-index.jpeg?raw=true "refractive_index")

The `refractive_index` property is a `float (0, inf]` that determines the index of refraction of the object. The index of refraction for empty space is set to 1.

To run the pictured scene:
```
python demo.py refractive_index.yaml --device [metal|cuda]
```

### Translucency
![Alt text](assets/translucency.jpeg?raw=true "translucency")

The `translucency` property is a `float [0, 1]` that determines how much the material scatters light that passes through it. A value of 0 is completely clear, 1 is completely clouded, and frankly anything greater than 0.1 is rather clouded.

To run the pictured scene:
```
python demo.py translucency.yaml --device [metal|cuda]
```

## Glossiness

Glossy surfaces have a chance of reflecting light in a way akin to a lacquer on a surface. `Glossiness` is a `bool`, and glossy surfaces give their color to the reflected ray if the ray passes through the gloss but reflect without giving color if the ray bounces off the glass. Glossy reflections are controlled by the `gloss_refractive_index` and `gloss_translucency` properties.

### Gloss Refractive Index
![Alt text](assets/gloss-refractive-index.jpeg?raw=true "gloss_refractive_index")

The `gloss_refractive_index` property is a `float (0, inf]` that determines the index of refraction of the glossy surface. This determines the probability that light will pass through or reflect off the gloss.

To run the pictured scene:
```
python demo.py gloss_refractive_index.yaml --device [metal|cuda]
```

### Gloss translucency
![Alt text](assets/gloss-translucency.jpeg?raw=true "gloss_translucency")

The `gloss_translucency` property is a `float [0, 1]` that determines how much the glossy surface scatters light that reflects off of it. This is the same as `translucency` but applied to glossy surfaces.

To run the pictured scene:
```
python demo.py gloss_translucency.yaml --device [metal|cuda]
```

# TODO
- Support for external mesh files
- Camera views inside triangle meshes should consider refraction (this currently works for spheres)
