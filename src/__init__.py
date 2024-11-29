from .scene import Scene
from .view import View
from .surfaces import Triangle, Sphere, Material
from .utils import load_yaml

default_save_config = {
    "save_path": "renders/default",
    "exposure": 3.0,
    "num_samples": 1000,
    "max_bounces": 500,
    "resolution": (3840, 2160),
}


def parse_yaml(yaml_path):
    scene_config = load_yaml(yaml_path)

    view_kwargs = scene_config["view"]
    view_kwargs["resolution"] = view_kwargs.pop("render_resolution")

    render_config = {}
    render_config["exposure"] = view_kwargs.pop("exposure")
    render_config["num_samples"] = view_kwargs.pop("num_samples")
    render_config["max_bounces"] = view_kwargs.pop("max_bounces")
    render_config["display_resolution"] = view_kwargs.pop("display_resolution")

    view = View(**view_kwargs)

    surfaces = []

    for surface in scene_config["surfaces"]:
        if "material" in surface:
            surface["material"] = Material(**surface["material"])
        type_ = surface["type"]
        del surface["type"]
        if type_ == "triangle":
            surfaces.append(Triangle(**surface))
        elif type_ == "sphere":
            surfaces.append(Sphere(**surface))

    scene = Scene(surfaces)

    save_config = scene_config.get("save_render", default_save_config)

    return scene, view, render_config, save_config
