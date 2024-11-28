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
  float3 normal;
  Material material;
};

__constant__ Hit NO_HIT = {INFINITY, {0, 0, 0}, {{0, 0, 0}, 0.0f, 0.0f}};

__device__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(const float3 &a, const float3 &b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
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

__device__ bool check_specular(float reflectivity, curandState *state) {
  return curand_uniform(state) < reflectivity;
}

__device__ float3 reflect_dir(float3 dir, float3 normal) {
  return dir - normal * (2.0f * dot(dir, normal));
}

__device__ float3 rand_dir(float3 normal, curandState *state) {
  float r1 = curand_normal(state);
  float r2 = curand_normal(state);
  float r3 = curand_normal(state);
  return normalize(normal + make_float3(r1, r2, r3));
}

__device__ float3 diffuse_dir(float3 normal, curandState *state) {
  float3 random_dir = rand_dir(normal, state);
  if (dot(random_dir, normal) < 0) {
    random_dir = make_float3(-random_dir.x, -random_dir.y, -random_dir.z);
  }
  return random_dir;
}

__device__ Ray add_environment(Ray ray) {
  float3 color;
  if (ray.dir.z > 0.98f) {
    float scale = (ray.dir.z - 0.98f) / 0.02f;
    color = WHITE * scale + SUN_COLOR * (1.0f - scale);
    ray.intensity += 1.0f;
  } else {
    color = SKY_COLOR;
    ray.intensity += 0.5f;
  }
  color = SKY_COLOR;
  ray.intensity += 0.5f;
  ray.color = ray.color * color;
  return ray;
}

__device__ Hit sphere_hit(Ray ray, Sphere sphere) {
  float3 ray_offset_origin = ray.origin - sphere.center;
  float b = 2.0f * dot(ray.dir, ray_offset_origin);
  float c =
      dot(ray_offset_origin, ray_offset_origin) - sphere.radius * sphere.radius;
  float discriminant = b * b - 4.0f * c;

  if (discriminant > 0) {
    float t = (-b - sqrtf(discriminant)) / 2.0f;
    if (t > 0) {
      return Hit{
        t, 
        normalize(ray.origin + ray.dir * t - sphere.center) / sphere.radius), 
        sphere.material
      };
    }
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

  return Hit{t, normalize(cross(ab, ac)), triangle.material};
}

__device__ Ray get_ray(const View view, int x, int y, curandState *state) {
  float x_offset = static_cast<float>(x) + curand_uniform(state) - 0.5f;
  float y_offset = static_cast<float>(y) + curand_uniform(state) - 0.5f;

  Ray ray;
  ray.origin = view.origin;
  ray.dir = normalize(view.top_left_dir + view.right_dir * x_offset +
                      view.down_dir * y_offset);
  ray.color = make_float3(1.0f, 1.0f, 1.0f);
  ray.intensity = 0.0f;
  return ray;
}

extern "C" {

__global__ void init_rand_state(curandState *states, int seed, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx > size)
    return;
  curand_init(seed, idx, 0, &states[idx]);
}

__global__ void trace_rays(View *view, Sphere *spheres, Triangle *triangles,
                           int num_spheres, int num_triangles, int num_bounces,
                           int num_rays, float exposure,
                           curandState *rand_states, float3 *image) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= view->width * view->height)
    return;

  int x = idx % view->width;
  int y = idx / view->width;
  curandState *local_state = &rand_states[idx];

  float3 pixel_color = make_float3(0.0f, 0.0f, 0.0f);

  for (int ray_num = 0; ray_num < num_rays; ray_num++) {
    Ray ray = get_ray(*view, x, y, local_state);

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
        if (check_specular(closest_hit.material.reflectivity, local_state)) {
          ray.dir = reflect_dir(ray.dir, closest_hit.normal);
        } else {
          ray.color = ray.color * closest_hit.material.color;
          ray.intensity += closest_hit.material.intensity;
          ray.dir = diffuse_dir(closest_hit.normal, local_state);
        }
      } else {
        ray = add_environment(ray);
        break;
      }
    }
    pixel_color = pixel_color + ray.color * ray.intensity;
  }

  pixel_color = pixel_color / static_cast<float>(num_rays);

  float3 ldr;
  ldr.x = (1.0f - expf(-pixel_color.x * exposure)) * 255.0f;
  ldr.y = (1.0f - expf(-pixel_color.y * exposure)) * 255.0f;
  ldr.z = (1.0f - expf(-pixel_color.z * exposure)) * 255.0f;

  image[idx] = ldr;
}
}
