import numpy as np

from triangle import Triangle
from ray import Ray

triangle = Triangle(
    points=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    color=np.array([0, 0, 0]),
    material="diffuse",
)

triangle2 = Triangle(
    points=np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
    color=np.array([0, 0, 0]),
    material="diffuse",
)
ray = Ray(
    origin=np.array([0.0, 0.0, 0]),
    dir=np.array([1, 1, 1]),
    pixel_coords=(0, 0),
)

print(triangle.check_hit(ray))
print(triangle2.check_hit(ray))
