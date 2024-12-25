#include <metal_stdlib>

using namespace metal;

constant float EPS = 1e-6;
constant float BIG_EPS = 1e-3;
constant float AIR_REF_INDEX = 1.0;
constant int MAX_STACK_SIZE = 8;

class SimpleRNG {
private:
  thread uint state;

public:
  thread SimpleRNG(const unsigned seed1, const unsigned seed2 = 1) {
    this->state = seed1 * 1103515245 + seed2 * 4928004 / seed1;
  }
  thread float rand() {
    this->state = (this->state + 592831) * 1103515245 + 12345;
    return float(this->state & 0x7FFFFFFF) / float(0x7FFFFFFF);
  }
  thread float rand_normal() {
    // mean is 0; standard deviation is 1
    return sqrt(-2.0 * log(this->rand())) * sin(2.0 * M_PI_F * this->rand());
  }
};

struct Ray {
  packed_float3 origin;
  packed_float3 dir;
  packed_float3 color;
  float intensity;
};

struct View {
  packed_float3 origin;
  packed_float3 top_left_dir;
  packed_float3 right_dir;
  packed_float3 down_dir;
  int width;
  int height;
};

struct Material {
  packed_float3 color;
  float intensity;
  float reflectivity;
  int transparent;
  float refractive_index;
  float translucency;
  packed_float3 absorption;
  int glossy;
  float gloss_refractive_index;
  float gloss_translucency;
};

struct Sphere {
  packed_float3 center;
  float radius;
  Material material;
};

struct Triangle {
  packed_float3 v0;
  packed_float3 ab;
  packed_float3 ac;
  packed_float3 normal;
  Material material;
  int mesh_id;
};

struct Hit {
  float t;
  bool internal;
  packed_float3 normal;
  Material material;
  int mesh_id;
};

struct Mesh {
  Material material;
  int mesh_id;
};

template <int N> struct MeshStack {
  Mesh data[N];
  int top = 0;

  void push(thread Mesh &item) {
    if (top < N) {
      data[top++] = item;
    }
  }

  Mesh pop() {
    if (top > 0) {
      return data[--top];
    }
    Mesh v;
    v.mesh_id = -1;
    return v;
  }

  Mesh peek() {
    if (top > 0) {
      return data[top - 1];
    }
    Mesh v;
    v.mesh_id = -1;
    return v;
  }

  bool empty() { return top == 0; }

  bool full() { return top == N; }
};

constant Hit NO_HIT =
    Hit{INFINITY, false, packed_float3(0.0f, 0.0f, 0.0f),
        Material{packed_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f}, -1};

float schlick_fresnel(float cosine, float eta1, float eta2) {
  float r0 = (eta1 - eta2) / (eta1 + eta2);
  r0 = r0 * r0;
  return r0 + (1 - r0) * pow((1 - cosine), 5);
}

bool check_transmission(float eta1, float eta2, packed_float3 dir,
                        packed_float3 normal, thread SimpleRNG &rng) {
  float reflect_prob = schlick_fresnel(abs(dot(dir, normal)), eta1, eta2);
  return rng.rand() > reflect_prob;
}

packed_float3 rand_dir(packed_float3 normal, thread SimpleRNG &rng) {
  float r1 = rng.rand_normal();
  float r2 = rng.rand_normal();
  float r3 = rng.rand_normal();
  return normalize(normal + packed_float3(r1, r2, r3));
}

packed_float3 reflect_diffuse(packed_float3 normal, thread SimpleRNG &rng) {
  packed_float3 random_dir = rand_dir(normal, rng);
  if (dot(random_dir, normal) <= 0) {
    random_dir = -random_dir;
  }
  return random_dir;
}
packed_float3 reflect_specular(packed_float3 dir, packed_float3 normal) {
  return dir - 2 * dot(dir, normal) * normal;
}

packed_float3 refract_dir(packed_float3 dir, packed_float3 normal, float eta1,
                          float eta2, float translucency,
                          thread SimpleRNG &rng) {
  float cos_i = abs(dot(normal, dir));

  float ref_rat = eta1 / eta2;
  float cos_t_squared = 1.0f - ref_rat * ref_rat * (1.0f - cos_i * cos_i);

  if (cos_t_squared < 0.0f) {
    return reflect_specular(dir, normal);
  }

  packed_float3 refracted_dir = normalize(
      ref_rat * dir + (ref_rat * cos_i - sqrt(cos_t_squared)) * normal);
  packed_float3 diffuse_dir = reflect_diffuse(normal, rng);

  return normalize(mix(refracted_dir, diffuse_dir, translucency));
}
packed_float3 reflect(packed_float3 dir, packed_float3 normal,
                      float reflectivity, thread SimpleRNG &rng) {
  packed_float3 diffuse_dir = reflect_diffuse(normal, rng);
  packed_float3 specular_dir = reflect_specular(dir, normal);
  return normalize(mix(diffuse_dir, specular_dir, reflectivity));
}

packed_float3 attenuate(packed_float3 color, float t,
                        packed_float3 absorption) {
  return color * exp(-t * tan(M_PI_F / 2 * absorption));
}

packed_float3 tone_map(packed_float3 color, float exposure) {
  return (1 - exp(-color * exposure)) * 255.0;
}

Ray get_ray(constant View &view, uint base_id, thread SimpleRNG &rng) {
  float x_offset =
      static_cast<float>(base_id % view.width) + 3 * rng.rand() - 1.5f;
  int y_offset =
      static_cast<float>(base_id / view.width) + 3 * rng.rand() - 1.5f;

  return Ray{view.origin,
             normalize(view.top_left_dir + x_offset * view.right_dir +
                       y_offset * view.down_dir),
             packed_float3(1.0f, 1.0f, 1.0f), 0.0f};
}

Hit sphere_hit(Ray ray, Sphere sphere) {
  packed_float3 offset = ray.origin - sphere.center;
  float b = 2 * dot(ray.dir, offset);
  float c = length_squared(offset) - sphere.radius * sphere.radius;
  float discriminant = b * b - 4 * c;

  if (discriminant > 0) {
    float sqrt_d = sqrt(discriminant);
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

    packed_float3 hit_point = ray.origin + t * ray.dir;
    packed_float3 normal = normalize(hit_point - sphere.center);

    if (internal) {
      normal = -normal;
    }

    return Hit{t, internal, normal, sphere.material, -1};
  }
  return NO_HIT;
}

Hit triangle_hit(Ray ray, Triangle triangle) {
  // checks both faces
  float3 pvec = cross(ray.dir, triangle.ac);
  float det = dot(triangle.ab, pvec);

  if (abs(det) < EPS) {
    return NO_HIT;
  }

  float inv_det = 1.0 / (det + EPS);
  float3 tvec = ray.origin - triangle.v0;
  float u = dot(tvec, pvec) * inv_det;
  if (u < 0.0 || u > 1.0) {
    return NO_HIT;
  }

  float3 qvec = cross(tvec, triangle.ab);
  float v = dot(ray.dir, qvec) * inv_det;
  if (v < 0.0 || u + v > 1.0) {
    return NO_HIT;
  }

  float t = dot(triangle.ac, qvec) * inv_det;
  if (t < 10 * EPS) {
    return NO_HIT;
  }

  float3 normal = triangle.normal;
  if (det < 0) {
    normal = -normal;
  }

  return Hit{t, false, normal, triangle.material, triangle.mesh_id};
}

MeshStack<MAX_STACK_SIZE>
get_surrounding_media(const device Sphere *spheres,
                      const device int *surrounding_spheres,
                      constant int &num_surrounding_spheres) {
  MeshStack<MAX_STACK_SIZE> media_stack;
  for (int i = num_surrounding_spheres - 1; i >= 0; i--) {
    Material material = spheres[surrounding_spheres[i]].material;
    Mesh item = {material, -1};
    media_stack.push(item);
  }
  return media_stack;
}

kernel void trace_rays(constant View &view [[buffer(0)]],
                       constant int &seed [[buffer(1)]],
                       const device Sphere *spheres [[buffer(2)]],
                       const device Triangle *triangles [[buffer(3)]],
                       const device int *surrounding_spheres [[buffer(4)]],
                       constant int &num_spheres [[buffer(5)]],
                       constant int &num_triangles [[buffer(6)]],
                       constant int &num_surrounding_spheres [[buffer(7)]],
                       constant int &num_bounces [[buffer(8)]],
                       constant int &num_rays [[buffer(9)]],
                       const device packed_float3 *background_color [[buffer(10)]],
                       constant float &background_luminance [[buffer(11)]],
                       constant float &exposure [[buffer(12)]],
                       constant bool &accumulate [[buffer(13)]],
                       constant int &iteration [[buffer(14)]],
                       device packed_float3 *accumulation [[buffer(15)]],
                       device packed_float3 *out [[buffer(16)]],
                       uint id [[thread_position_in_grid]]) {
  SimpleRNG rng = SimpleRNG(seed, id * id);

  MeshStack<MAX_STACK_SIZE> _inside_stack = get_surrounding_media(
      spheres, surrounding_spheres, num_surrounding_spheres);

  packed_float3 pixel = packed_float3(0.0f, 0.0f, 0.0f);

  for (int ray_num = 0; ray_num < num_rays; ray_num++) {
    MeshStack<MAX_STACK_SIZE> inside_stack = _inside_stack;
    Ray ray = get_ray(view, id, rng);

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
                              1 - closest_hit.material.translucency, rng);
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
        ray.intensity = background_luminance;
        ray.color = ray.color * background_color[0];
        break;
      }
    }

    pixel += ray.color * ray.intensity;
  }

  pixel = pixel / num_rays;

  if (accumulate) {
    if (iteration == 0) {
      accumulation[id] = pixel;
    } else {
      accumulation[id] += pixel;
    }

    packed_float3 avg = accumulation[id] / (iteration + 1);
    out[id] = tone_map(avg, exposure);

  } else {
    out[id] = tone_map(pixel, exposure);
  }
}
