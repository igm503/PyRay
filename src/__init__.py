from .scene import Scene
from .view import View
from .surfaces import Triangle, Sphere, Material
from .utils import load_yaml

__all__ = ["parse_yaml"]


def parse_yaml(yaml_path):
    scene_config = load_yaml(yaml_path)

    view_kwargs = scene_config["view"]
    view = View(**view_kwargs)

    surfaces = []

    for surface in scene_config["surfaces"]:
        print(surface)
        if "material" in surface:
            surface["material"] = Material(**surface["material"])
        type_ = surface["type"]
        del surface["type"]
        if type_ == "triangle":
            surfaces.append(Triangle(**surface))
        elif type_ == "sphere":
            surfaces.append(Sphere(**surface))

    scene = Scene(surfaces)
    return scene, view
