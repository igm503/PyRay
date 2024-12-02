#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>

#define EPSILON 1e-6f

__constant__ float3 SUN_COLOR = {1.0f, 0.68f, 0.26f};
__constant__ float3 WHITE = {1.0f, 1.0f, 1.0f};
__constant__ float3 SKY_COLOR = {0.53f, 0.81f, 0.92f};

struct Ray {
  float3 origin;
  float3 dir;
  float3 color;
  float intensity;
};

struct View {
  float3 origin;
  float3 top_left_dir;
  float3 right_dir;
  float3 down_dir;
  int width;
  int height;
};

struct Material {
  float3 color;
  float intensity;
  float reflectivity;
  float transparency;
  float translucency;
  float refractive_index;
};

struct Sphere {
  float3 center;
  float radius;
  Material material;
};

struct Triangle {
  float3 v0;
  float3 v1;
  float3 v2;
  Material material;
};

struct Hit {
  float t;
  bool internal;
  float3 normal;
  Material material;
};

__constant__ Hit NO_HIT = {INFINITY, false, {0, 0, 0}, {{0, 0, 0}, 0.0f, 0.0f}};

// vector math

__device__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(const float3 &a, const float3 &b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ float3 operator*(const float &a, const float3 &b) {
  return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ float3 operator*(const float3 &a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator/(const float3 &a, float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ float dot(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 cross(const float3 &a, const float3 &b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__device__ float3 normalize(float3 v) {
  float len = sqrtf(dot(v, v));
  return v / len;
}

__device__ float3 lerp(const float3 &a, const float3 &b, const float &w) {
  return a * (1.0f - w) + b * w;
}

// reflection functions

__device__ float schlick_fresnel(float cosine, float eta1, float eta2) {
  float r0 = (eta1 - eta2) / (eta1 + eta2);
  r0 = r0 * r0;
  return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool check_transmission(float transparency, float eta1, float eta2,
                                   float3 dir, float3 normal,
                                   curandState *state) {
  if (transparency <= 0.0f) {
    return false;
  }
  float fresnel = schlick_fresnel(abs(dot(dir, normal)), eta1, eta2);
  float reflection_prob = fresnel * (1.0f - transparency);

  return curand_uniform(state) < reflection_prob;
}

__device__ float3 rand_dir(float3 normal, curandState *state) {
  float r1 = curand_normal(state);
  float r2 = curand_normal(state);
  float r3 = curand_normal(state);
  return normalize(normal + make_float3(r1, r2, r3));
}

__device__ float3 reflect_diffuse(float3 normal, curandState *state) {
  float3 random_dir = rand_dir(normal, state);
  if (dot(random_dir, normal) < 0) {
    random_dir = make_float3(-random_dir.x, -random_dir.y, -random_dir.z);
  }
  return random_dir;
}

__device__ float3 reflect_specular(float3 dir, float3 normal) {
  return dir - normal * (2.0f * dot(dir, normal));
}

__device__ float3 refract_dir(float3 dir, float3 normal, bool internal,
                              float ref_rat, float translucency,
                              curandState *state) {

  dir = normalize(dir);
  normal = normalize(normal);

  float cos_i = dot(normal, dir);

  float cos_t_squared = 1.0f - ref_rat * ref_rat * (1.0f - cos_i * cos_i);

  if (cos_t_squared < 0.0f) {
    return reflect_specular(dir, normal);
  }

  float3 refracted_dir = normalize(
      ref_rat * dir + (ref_rat * cos_i - sqrt(cos_t_squared)) * normal);
  float3 diffuse_dir = reflect_diffuse(normal, state);

  return normalize(lerp(refracted_dir, diffuse_dir, translucency));
}

__device__ float3 reflect(float3 dir, float3 normal, float reflectivity,
                          curandState *state) {
  float3 diffuse_dir = reflect_diffuse(normal, state);
  float3 specular_dir = reflect_specular(dir, normal);
  return normalize(lerp(diffuse_dir, specular_dir, reflectivity));
}

__device__ float3 tone_map(float3 color, float exposure) {
  float3 ldr;
  ldr.x = (1.0f - expf(-color.x * exposure)) * 255.0f;
  ldr.y = (1.0f - expf(-color.y * exposure)) * 255.0f;
  ldr.z = (1.0f - expf(-color.z * exposure)) * 255.0f;
  return ldr;
}

__device__ Ray add_environment(Ray ray) {
  ray.intensity = 0.5f;
  ray.color = ray.color * SKY_COLOR;
  return ray;
}

__device__ Ray get_ray(const View view, int idx, curandState *state) {
  float x_offset =
      static_cast<float>(idx % view.width) + 3 * curand_uniform(state) - 1.5f;
  float y_offset =
      static_cast<float>(idx / view.width) + 3 * curand_uniform(state) - 1.5f;

  return Ray{view.origin,
             normalize(view.top_left_dir + view.right_dir * x_offset +
                       view.down_dir * y_offset),
             make_float3(1.0f, 1.0f, 1.0f), 0.0f};
}

__device__ Hit sphere_hit(Ray ray, Sphere sphere) {
  float3 offset = ray.origin - sphere.center;
  float b = 2.0f * dot(ray.dir, offset);
  float c = dot(offset, offset) - sphere.radius * sphere.radius;
  float discriminant = b * b - 4.0f * c;

  if (discriminant > 0) {
    float sqrt_d = sqrtf(discriminant);
    float t1 = (-b - sqrt_d) / 2.0f;
    float t2 = (-b + sqrt_d) / 2.0f;

    float t;
    bool internal;
    if (t1 > EPSILON) {
      t = t1;
      internal = false;
    } else if (t2 > EPSILON && sphere.material.transparency > 0.0f) {
      t = t2;
      internal = true;
    } else {
      return NO_HIT;
    }

    float3 hit_point = ray.origin + t * ray.dir;
    float3 normal = normalize(hit_point - sphere.center);

    if (internal) {
      normal = make_float3(-normal.x, -normal.y, -normal.z);
    }
    return Hit{t, internal, normal, sphere.material};
  }
  return NO_HIT;
}

__device__ Hit triangle_hit(Ray ray, Triangle triangle) {
  float3 ab = triangle.v1 - triangle.v0;
  float3 ac = triangle.v2 - triangle.v0;
  float3 pvec = cross(ray.dir, ac);
  float det = dot(ab, pvec);

  if (det < EPSILON) {
    return NO_HIT;
  }

  float inv_det = 1.0f / (det + EPSILON);
  float3 tvec = ray.origin - triangle.v0;
  float u = dot(tvec, pvec) * inv_det;
  if (u < 0.0f || u > 1.0f) {
    return NO_HIT;
  }

  float3 qvec = cross(tvec, ab);
  float v = dot(ray.dir, qvec) * inv_det;
  if (v < 0.0f || u + v > 1.0f) {
    return NO_HIT;
  }

  float t = dot(ac, qvec) * inv_det;
  if (t < 10.0f * EPSILON) {
    return NO_HIT;
  }

  float3 normal = normalize(cross(ab, ac));
  if (det < 0.0f) {
    normal = make_float3(-normal.x, -normal.y, -normal.z);
  }
  return Hit{t, false, normal, triangle.material};
}

extern "C" {

__global__ void init_rand_state(curandState *states, int seed, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx > size)
    return;
  curand_init(seed, idx, 0, &states[idx]);
}

__global__ void trace_rays(View *view, curandState *rand_states,
                           Sphere *spheres, Triangle *triangles,
                           int num_spheres, int num_triangles, int num_bounces,
                           int num_rays, float exposure, int accumulate,
                           int iteration, float3 *accumulation, float3 *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= view->width * view->height)
    return;

  curandState *local_state = &rand_states[idx];

  float3 pixel = make_float3(0.0f, 0.0f, 0.0f);

  for (int ray_num = 0; ray_num < num_rays; ray_num++) {
    Ray ray = get_ray(*view, idx, local_state);

    for (int bounce = 0; bounce < num_bounces; bounce++) {
      Hit closest_hit = NO_HIT;

      for (int i = 0; i < num_spheres; i++) {
        Sphere sphere = spheres[i];
        Hit hit = sphere_hit(ray, sphere);
        if (hit.t < closest_hit.t) {
          closest_hit = hit;
        }
      }

      for (int i = 0; i < num_triangles; i++) {
        Triangle triangle = triangles[i];
        Hit hit = triangle_hit(ray, triangle);
        if (hit.t < closest_hit.t) {
          closest_hit = hit;
        }
      }

      if (closest_hit.t < INFINITY) {

        ray.origin = ray.origin + ray.dir * closest_hit.t;
        ray.color = ray.color * closest_hit.material.color;

        if (closest_hit.material.intensity > 0) {
          ray.intensity = closest_hit.material.intensity;
          break;
        }

        float eta1;
        float eta2;
        if (closest_hit.internal) {
          eta1 = closest_hit.material.refractive_index;
          eta2 = 1.0f;
        } else {
          eta1 = 1.0f;
          eta2 = closest_hit.material.refractive_index;
        }
        float ref_rat = eta1 / eta2;
        bool is_transmission =
            check_transmission(closest_hit.material.transparency, eta1, eta2,
                               ray.dir, closest_hit.normal, local_state);
        if (is_transmission) {
          ray.origin = ray.origin - 100 * EPSILON * closest_hit.normal;
          ray.dir = refract_dir(ray.dir, closest_hit.normal,
                                closest_hit.internal, ref_rat,
                                closest_hit.material.translucency, local_state);
        } else {
          ray.dir = reflect(ray.dir, closest_hit.normal,
                            closest_hit.material.reflectivity, local_state);
        }
      } else {
        ray = add_environment(ray);
        break;
      }
    }
    pixel = pixel + ray.color * ray.intensity;
  }

  pixel = pixel / static_cast<float>(num_rays);

  if (accumulate == 1) {
    if (iteration == 0) {
      accumulation[idx] = pixel;
    } else {
      accumulation[idx] = accumulation[idx] + pixel;
    }

    float3 avg = accumulation[idx] / (iteration + 1);
    out[idx] = tone_map(avg, exposure);

  } else {
    out[idx] = tone_map(pixel, exposure);
  }
}
}
