from dataclasses import dataclass

import numpy as np

from ..view import View
from ..ray import Ray, Hit
from ..surfaces import Material, Triangle, Sphere


EPS = 1e-6
BIG_EPS = 1e-3
AIR_REF_INDEX = 1.0
MAX_STACK_SIZE = 8

rng = np.random.default_rng(0)


@dataclass
class Mesh:
    material: Material
    mesh_id: int


class CPUTracer:
    def render_iteration(
        self,
        view: View,
        spheres: list[Sphere],
        triangles: list[Triangle],
        surrounding_spheres: list[int],
        num_rays: int,
        max_bounces: int,
        background_color: tuple[float, float, float],
        background_luminance: float,
        exposure: float,
        accumulate: bool,
        iteration: int = 0,
    ):
        img = trace_rays(
            view,
            spheres,
            triangles,
            surrounding_spheres,
            max_bounces,
            num_rays,
            np.array(background_color, dtype=np.float32),
            background_luminance,
            exposure,
            accumulate,
            iteration,
        )
        return img.astype(np.uint8)


def schlick_fresnel(cosine: float, eta1: float, eta2: float) -> float:
    r0 = (eta1 - eta2) / (eta1 + eta2)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)


def check_transmission(eta1: float, eta2: float, dir: np.ndarray, normal: np.ndarray) -> bool:
    reflect_prob = schlick_fresnel(abs(np.dot(dir, normal)), eta1, eta2)
    return rng.random() > reflect_prob


def rand_dir(normal: np.ndarray) -> np.ndarray:
    r1, r2, r3 = rng.standard_normal(3)
    random_dir = normal + np.array([r1, r2, r3], dtype=np.float32)
    return random_dir / np.linalg.norm(random_dir)


def reflect_diffuse(normal: np.ndarray) -> np.ndarray:
    random_dir = rand_dir(normal)
    if np.dot(random_dir, normal) <= 0:
        random_dir = -random_dir
    return random_dir


def reflect_specular(dir: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return dir - 2 * np.dot(dir, normal) * normal


def refract_dir(
    dir: np.ndarray, normal: np.ndarray, eta1: float, eta2: float, translucency: float
) -> np.ndarray:
    cos_i = abs(np.dot(normal, dir))

    ref_rat = eta1 / eta2
    cos_t_squared = 1.0 - ref_rat * ref_rat * (1.0 - cos_i * cos_i)

    if cos_t_squared < 0.0:
        return reflect_specular(dir, normal)

    refracted_dir = ref_rat * dir + (ref_rat * cos_i - np.sqrt(cos_t_squared)) * normal
    refracted_dir = refracted_dir / np.linalg.norm(refracted_dir)

    diffuse_dir = reflect_diffuse(normal)

    # Linear interpolation between refracted and diffuse direction
    final_dir = (1 - translucency) * refracted_dir + translucency * diffuse_dir
    return final_dir / np.linalg.norm(final_dir)


def reflect(dir: np.ndarray, normal: np.ndarray, reflectivity: float) -> np.ndarray:
    diffuse_dir = reflect_diffuse(normal)
    specular_dir = reflect_specular(dir, normal)

    final_dir = (1 - reflectivity) * diffuse_dir + reflectivity * specular_dir
    return final_dir / np.linalg.norm(final_dir)


def attenuate(color: np.ndarray, t: float, absorption: np.ndarray) -> np.ndarray:
    return color * np.exp(-t * np.tan(np.pi / 2 * absorption))


def tone_map(color: np.ndarray, exposure: float) -> np.ndarray:
    return (1 - np.exp(-color * exposure)) * 255.0


def get_ray(view: View, base_id: int) -> Ray:
    x_offset = float(base_id % view.width) + 3 * rng.random() - 1.5
    y_offset = float(base_id // view.width) + 3 * rng.random() - 1.5

    dir = view.top_left_dir + x_offset * view.right_dir + y_offset * view.down_dir
    dir = dir / np.linalg.norm(dir)

    return Ray(origin=view.origin, dir=dir)


def sphere_hit(ray: Ray, sphere: Sphere) -> Hit | None:
    offset = ray.origin - sphere.center
    b = 2 * np.dot(ray.dir, offset)
    c = np.dot(offset, offset) - sphere.radius * sphere.radius
    discriminant = b * b - 4 * c

    if discriminant > 0:
        sqrt_d = np.sqrt(discriminant)
        t1 = (-b - sqrt_d) / 2.0
        t2 = (-b + sqrt_d) / 2.0

        if t1 > EPS:
            t = t1
            internal = False
        elif t2 > EPS:
            t = t2
            internal = True
        else:
            return None

        hit_point = ray.origin + t * ray.dir
        normal = (hit_point - sphere.center) / np.linalg.norm(hit_point - sphere.center)

        if internal:
            normal = -normal

        return Hit(t=t, internal=internal, normal=normal, material=sphere.material, mesh_id=-1)

    return None


def triangle_hit(ray: Ray, triangle: Triangle) -> Hit | None:
    pvec = np.cross(ray.dir, triangle.ac)
    det = np.dot(triangle.ab, pvec)

    if abs(det) < EPS:
        return None

    inv_det = 1.0 / (det + EPS)
    tvec = ray.origin - triangle.v0
    u = np.dot(tvec, pvec) * inv_det

    if u < 0.0 or u > 1.0:
        return None

    qvec = np.cross(tvec, triangle.ab)
    v = np.dot(ray.dir, qvec) * inv_det

    if v < 0.0 or u + v > 1.0:
        return None

    t = np.dot(triangle.ac, qvec) * inv_det

    if t < 10 * EPS:
        return None

    normal = triangle.normal

    if det < 0:
        normal = -normal

    return Hit(
        t=t,
        internal=False,
        normal=normal,
        material=triangle.material,
        mesh_id=triangle.mesh_id,
    )


def get_surrounding_media(
    spheres: list[Sphere], surrounding_sphere_indices: list[int]
) -> list[Mesh]:
    media = []
    for idx in reversed(surrounding_sphere_indices):
        material = spheres[idx].material
        media.append(Mesh(material=material, mesh_id=-1))
    return media


def trace_rays(
    view: View,
    spheres: list[Sphere],
    triangles: list[Triangle],
    surrounding_sphere_indices: list[int],
    num_bounces: int,
    num_rays: int,
    background_color: np.ndarray,
    background_luminance: float,
    exposure: float,
    accumulate: bool,
    iteration: int,
) -> np.ndarray:
    width, height = view.width, view.height
    output = np.zeros((height, width, 3), dtype=np.float32)

    if accumulate:
        accumulation = np.zeros((height, width, 3), dtype=np.float32)

    _inside_stack = get_surrounding_media(spheres, surrounding_sphere_indices)

    for y in range(height):
        for x in range(width):
            pixel_id = y * width + x
            pixel_color = np.zeros(3, dtype=np.float32)

            for ray_num in range(num_rays):
                inside_stack = _inside_stack.copy()

                ray = get_ray(view, pixel_id)

                for bounce in range(num_bounces):
                    closest_hit = None
                    closest_t = float("inf")

                    for sphere in spheres:
                        hit = sphere_hit(ray, sphere)
                        if hit and hit.t < closest_t:
                            closest_hit = hit
                            closest_t = hit.t

                    for triangle in triangles:
                        hit = triangle_hit(ray, triangle)
                        if hit and hit.t < closest_t:
                            closest_hit = hit
                            closest_t = hit.t

                    if closest_hit is None:
                        ray.intensity = background_luminance
                        ray.color = ray.color * background_color
                        break

                    if closest_hit.internal and not closest_hit.material.transparent:
                        break

                    hit_light = closest_hit.material.luminance > 0

                    if inside_stack:
                        ray.color = attenuate(
                            ray.color,
                            closest_hit.t,
                            inside_stack[-1].material.absorption,
                        )

                    if hit_light:
                        ray.color = ray.color * closest_hit.material.color
                        ray.intensity = closest_hit.material.luminance
                        break

                    if closest_hit.material.transparent:
                        if closest_hit.internal and len(inside_stack) == 1:
                            eta1 = closest_hit.material.refractive_index
                            eta2 = AIR_REF_INDEX
                        elif closest_hit.internal:
                            eta1 = closest_hit.material.refractive_index
                            eta2 = inside_stack[-1].material.refractive_index
                        elif inside_stack:
                            eta1 = inside_stack[-1].material.refractive_index
                            eta2 = closest_hit.material.refractive_index
                        else:
                            eta1 = AIR_REF_INDEX
                            eta2 = closest_hit.material.refractive_index

                        is_transmission = check_transmission(
                            eta1, eta2, ray.dir, closest_hit.normal
                        )

                        if is_transmission:
                            ray.origin = (
                                ray.origin + closest_hit.t * ray.dir - BIG_EPS * closest_hit.normal
                            )
                            if closest_hit.internal:
                                inside_stack.pop()
                            else:
                                if inside_stack:
                                    inside_volume = inside_stack[-1]
                                    prev_mesh_id = inside_volume.mesh_id
                                    if prev_mesh_id != -1 and prev_mesh_id == closest_hit.mesh_id:
                                        inside_stack.pop()
                                    else:
                                        inside_stack.append(
                                            Mesh(
                                                closest_hit.material,
                                                closest_hit.mesh_id,
                                            )
                                        )
                            ray.dir = refract_dir(
                                ray.dir,
                                closest_hit.normal,
                                eta1,
                                eta2,
                                closest_hit.material.translucency,
                            )
                        else:
                            ray.origin = (
                                ray.origin + closest_hit.t * ray.dir + BIG_EPS * closest_hit.normal
                            )
                            ray.dir = reflect(
                                ray.dir,
                                closest_hit.normal,
                                1 - closest_hit.material.translucency,
                            )
                    else:
                        reflectivity = closest_hit.material.reflectivity
                        if closest_hit.material.glossy:
                            is_transmission = check_transmission(
                                AIR_REF_INDEX,
                                closest_hit.material.gloss_refractive_index,
                                ray.dir,
                                closest_hit.normal,
                            )

                            if not is_transmission:
                                reflectivity = 1 - closest_hit.material.gloss_translucency
                            else:
                                ray.color = ray.color * closest_hit.material.color
                        else:
                            ray.color = ray.color * closest_hit.material.color

                        ray.origin = (
                            ray.origin + closest_hit.t * ray.dir + BIG_EPS * closest_hit.normal
                        )
                        ray.dir = reflect(ray.dir, closest_hit.normal, reflectivity)

                pixel_color += ray.color * ray.intensity

            pixel_color = pixel_color / num_rays

            if accumulate:
                if iteration == 0:
                    accumulation[y, x] = pixel_color
                else:
                    accumulation[y, x] += pixel_color

                avg = accumulation[y, x] / (iteration + 1)
                output[y, x] = tone_map(avg, exposure)
            else:
                output[y, x] = tone_map(pixel_color, exposure)

    return output
