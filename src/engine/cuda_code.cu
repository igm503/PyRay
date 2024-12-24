#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>

#define EPS 1e-6f
#define BIG_EPS 1e-3f
#define MAX_STACK_SIZE 8

__constant__ float3 SKY_COLOR = {0.53f, 0.81f, 0.92f};
__constant__ float AIR_REF_INDEX = 1.0;

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
  int transparent;
  float refractive_index;
  float translucency;
  float3 absorption;
  int glossy;
  float gloss_refractive_index;
  float gloss_translucency;
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
  int mesh_id;
};

struct Hit {
  float t;
  bool internal;
  float3 normal;
  Material material;
  int mesh_id;
};

struct Mesh {
  Material material;
  int mesh_id;
};

template <int N> struct MeshStack {
  Mesh *data;
  int top;

  __device__ MeshStack() {
    top = 0;
    data = new Mesh[N];
  }

  __device__ ~MeshStack() { delete[] data; }

  __device__ void push(const Mesh &item) {
    if (top < N) {
      data[top++] = item;
    }
  }

  __device__ Mesh pop() {
    if (top > 0) {
      return data[--top];
    }
    Mesh v;
    v.mesh_id = -1;
    return v;
  }

  __device__ Mesh peek() {
    if (top > 0) {
      return data[top - 1];
    }
    Mesh v;
    v.mesh_id = -1;
    return v;
  }

  __device__ bool empty() { return top == 0; }
  __device__ bool full() { return top == N; }
};

__constant__ Hit NO_HIT = {
    INFINITY, false, {0, 0, 0}, {{0, 0, 0}, 0.0f, 0.0f}, -1};

// vector math

__device__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 &operator+=(float3 &a, const float3 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator-(float a, const float3 &b) {
  return make_float3(a - b.x, a - b.y, a - b.z);
}

__device__ float3 operator-(const float3 &a) {
  return make_float3(-a.x, -a.y, -a.z);
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

__device__ float3 expf(const float3 &a) {
  return make_float3(expf(a.x), expf(a.y), expf(a.z));
}

__device__ float3 tan(const float3 &a) {
  return make_float3(tanf(a.x), tanf(a.y), tanf(a.z));
}

// reflection functions

__device__ float schlick_fresnel(float cosine, float eta1, float eta2) {
  float r0 = (eta1 - eta2) / (eta1 + eta2);
  r0 = r0 * r0;
  return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool check_transmission(float eta1, float eta2, float3 dir,
                                   float3 normal, curandState *state) {
  float reflect_prob = schlick_fresnel(abs(dot(dir, normal)), eta1, eta2);
  return curand_uniform(state) > reflect_prob;
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
    random_dir = -random_dir;
  }
  return random_dir;
}

__device__ float3 reflect_specular(float3 dir, float3 normal) {
  return dir - normal * (2.0f * dot(dir, normal));
}

__device__ float3 refract_dir(float3 dir, float3 normal, float eta1, float eta2,
                              float translucency, curandState *state) {

  float cos_i = abs(dot(normal, dir));

  float ref_rat = eta1 / eta2;
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

__device__ float3 attenuate(float3 color, float t, float3 absorption) {
  return color * expf(-t * tan(CUDART_PIO2_F * absorption));
}

__device__ float3 tone_map(float3 color, float exposure) {
  return 255.0f * (1.0f - expf(-color * exposure));
}

__device__ Ray add_environment(Ray ray) {
  ray.intensity = 1.0f;
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
    if (t1 > EPS) {
      t = t1;
      internal = false;
    } else if (t2 > EPS) {
      t = t2;
      internal = true;
    } else {
      return NO_HIT;
    }

    float3 hit_point = ray.origin + t * ray.dir;
    float3 normal = normalize(hit_point - sphere.center);

    if (internal) {
      normal = -normal;
    }
    return Hit{t, internal, normal, sphere.material, -1};
  }
  return NO_HIT;
}

__device__ Hit triangle_hit(Ray ray, Triangle triangle) {
  float3 ab = triangle.v1 - triangle.v0;
  float3 ac = triangle.v2 - triangle.v0;
  float3 pvec = cross(ray.dir, ac);
  float det = dot(ab, pvec);

  if (abs(det) < EPS) {
    return NO_HIT;
  }

  float inv_det = 1.0f / (det + EPS);
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
  if (t < 10.0f * EPS) {
    return NO_HIT;
  }

  float3 normal = normalize(cross(ab, ac));
  if (det < 0.0f) {
    normal = -normal;
  }
  return Hit{t, false, normal, triangle.material, triangle.mesh_id};
}

__device__ MeshStack<MAX_STACK_SIZE>
get_surrounding_media(const Sphere *spheres, const int *surrounding_spheres,
                      const int num_surrounding_spheres) {
  MeshStack<MAX_STACK_SIZE> media_stack;
  for (int i = num_surrounding_spheres - 1; i >= 0; i--) {
    Material material = spheres[surrounding_spheres[i]].material;
    Mesh item = {material, -1};
    media_stack.push(item);
  }
  return media_stack;
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
                           int *surrounding_spheres, int num_spheres,
                           int num_triangles, int num_surrounding_spheres,
                           int num_bounces, int num_rays, float exposure,
                           int accumulate, int iteration, float3 *accumulation,
                           float3 *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= view->width * view->height)
    return;

  curandState *rng = &rand_states[idx];

  MeshStack<MAX_STACK_SIZE> _inside_stack = get_surrounding_media(
      spheres, surrounding_spheres, num_surrounding_spheres);

  float3 pixel = make_float3(0.0f, 0.0f, 0.0f);

  for (int ray_num = 0; ray_num < num_rays; ray_num++) {
    MeshStack<MAX_STACK_SIZE> inside_stack = _inside_stack;
    Ray ray = get_ray(*view, idx, rng);

    for (int bounce = 0; bounce < num_bounces; bounce++) {
      Hit closest_hit = NO_HIT;

      for (int i = 0; i < num_spheres; i++) {
        Hit hit = sphere_hit(ray, spheres[i]);
        if (hit.t < closest_hit.t) {
          closest_hit = hit;
        }
      }

      for (int i = 0; i < num_triangles; i++) {
        Hit hit = triangle_hit(ray, triangles[i]);
        if (hit.t < closest_hit.t) {
          closest_hit = hit;
        }
      }

      if (closest_hit.t < INFINITY) {
        if (closest_hit.internal && closest_hit.material.transparent == 0) {
          break;
        }

        bool hit_light = closest_hit.material.intensity > 0;

        if (!inside_stack.empty()) {
          ray.color = attenuate(ray.color, closest_hit.t,
                                inside_stack.peek().material.absorption);
        }

        if (hit_light) {
          ray.color = ray.color * closest_hit.material.color;
          ray.intensity = closest_hit.material.intensity;
          break;
        }

        if (closest_hit.material.transparent == 1) {
          float eta1;
          float eta2;
          if (closest_hit.internal && inside_stack.top == 1) {
            eta1 = closest_hit.material.refractive_index;
            eta2 = AIR_REF_INDEX;
          } else if (closest_hit.internal) {
            eta1 = closest_hit.material.refractive_index;
            eta2 = inside_stack.peek().material.refractive_index;
          } else if (!inside_stack.empty()) {
            eta1 = inside_stack.peek().material.refractive_index;
            eta2 = closest_hit.material.refractive_index;
          } else {
            eta1 = AIR_REF_INDEX;
            eta2 = closest_hit.material.refractive_index;
          }
          bool is_transmission =
              check_transmission(eta1, eta2, ray.dir, closest_hit.normal, rng);
          if (is_transmission) {
            ray.origin = ray.origin + closest_hit.t * ray.dir -
                         BIG_EPS * closest_hit.normal;
            if (closest_hit.internal) {
              inside_stack.pop();
            } else {
              Mesh inside_volume = inside_stack.peek();
              int prev_mesh_id = inside_volume.mesh_id;
              if (prev_mesh_id != -1 && prev_mesh_id == closest_hit.mesh_id) {
                inside_stack.pop();
              } else {
                Mesh item = {closest_hit.material, closest_hit.mesh_id};
                inside_stack.push(item);
              }
            }
            ray.dir = refract_dir(ray.dir, closest_hit.normal, eta1, eta2,
                                  closest_hit.material.translucency, rng);
          } else {
            ray.origin = ray.origin + closest_hit.t * ray.dir +
                         BIG_EPS * closest_hit.normal;
            ray.dir = reflect(ray.dir, closest_hit.normal,
                              1.0f - closest_hit.material.translucency, rng);
          }
        }

        else {
          float reflectivity = closest_hit.material.reflectivity;
          if (closest_hit.material.glossy == 1) {
            bool is_transmission = check_transmission(
                AIR_REF_INDEX, closest_hit.material.gloss_refractive_index,
                ray.dir, closest_hit.normal, rng);

            if (!is_transmission) {
              reflectivity = 1 - closest_hit.material.gloss_translucency;
            } else {
              ray.color = ray.color * closest_hit.material.color;
            }
          } else {
            ray.color = ray.color * closest_hit.material.color;
          }
          ray.origin = ray.origin + closest_hit.t * ray.dir +
                       BIG_EPS * closest_hit.normal;
          ray.dir = reflect(ray.dir, closest_hit.normal, reflectivity, rng);
        }
      } else {
        ray = add_environment(ray);
        break;
      }
    }

    pixel += ray.color * ray.intensity;
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
