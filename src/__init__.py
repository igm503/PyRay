import numpy as np

from .scene import Scene
from .view import View
from .surfaces import Triangle, Sphere, Material
from .utils import load_yaml
from .constants import SKY_COLOR, SKY_LUMINANCE


def triangulate_quadrilateral(_points: list[list[float]]):
    points = np.array(_points)
    diagonal1_length = np.linalg.norm(points[0] - points[2])
    diagonal2_length = np.linalg.norm(points[1] - points[3])

    if diagonal1_length <= diagonal2_length:
        triangle1 = [points[0], points[1], points[2]]
        triangle2 = [points[2], points[3], points[0]]
    else:
        triangle1 = [points[1], points[2], points[3]]
        triangle2 = [points[3], points[0], points[1]]

    return (triangle1, triangle2)


def parse_surface_config(
    surface: dict, material: Material | None = None, mesh_id: int | None = None
):
    if material and "material" not in surface:
        surface["material"] = material
    else:
        surface["material"] = Material(**surface.get("material", {}))
    type_ = surface.pop("type")
    if mesh_id is not None:
        surface["mesh_id"] = mesh_id
    if type_ == "triangle":
        return [Triangle(**surface)]
    elif type_ == "sphere":
        return [Sphere(**surface)]
    elif type_ == "quad":
        points = surface.pop("points")
        points_1, points_2 = triangulate_quadrilateral(points)
        triangle1 = surface.copy()
        triangle1["points"] = points_1
        triangle2 = surface.copy()
        triangle2["points"] = points_2
        return [Triangle(**triangle1), Triangle(**triangle2)]
    elif type_ == "mesh":
        surfaces = surface.pop("surfaces")
        default_material = surface["material"]
        parsed_surfaces = [
            parse_surface_config(surface, default_material, mesh_id) for surface in surfaces
        ]
        return [surface for surfaces in parsed_surfaces for surface in surfaces]
    else:
        raise ValueError(f"Unknown surface type: {type_}")


def parse_yaml(yaml_path: str):
    scene_config = load_yaml(yaml_path)
    scene_name = yaml_path.split("/")[-1].split(".")[0]

    view_kwargs = scene_config["view"]
    view_kwargs["resolution"] = view_kwargs.pop("render_resolution")

    render_config = {}
    render_config["exposure"] = view_kwargs.pop("exposure")
    render_config["num_samples"] = view_kwargs.pop("num_samples")
    render_config["max_bounces"] = view_kwargs.pop("max_bounces")
    render_config["display_resolution"] = view_kwargs.pop("display_resolution")

    background_color = view_kwargs.pop("background_color", SKY_COLOR)
    background_luminance = view_kwargs.pop("background_luminance", SKY_LUMINANCE)

    view = View(**view_kwargs)

    surfaces = []

    current_mesh_id = 0
    for surface in scene_config["surfaces"]:
        if surface["type"] == "mesh":
            mesh_id = current_mesh_id
            current_mesh_id += 1
        else:
            mesh_id = None
        surfaces.extend(parse_surface_config(surface, mesh_id=mesh_id))

    scene = Scene(surfaces, background_color, background_luminance)

    save_kwargs = scene_config["save_render"]
    save_config = {}
    save_config["save_path"] = save_kwargs.pop("save_path", f"renders/{scene_name}")
    save_config["exposure"] = save_kwargs.pop("exposure", render_config["exposure"])
    save_config["num_samples"] = save_kwargs.pop("num_samples", 10000)
    save_config["max_bounces"] = save_kwargs.pop("max_bounces", 500)
    save_config["resolution"] = save_kwargs.pop("resolution", (3840, 2160))

    return scene, view, render_config, save_config
