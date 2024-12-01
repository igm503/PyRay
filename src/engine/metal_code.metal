#include <metal_stdlib>

using namespace metal;

constant float epsilon = 1e-6;

constant float3 sun_color = float3(1.0, 0.68, 0.26);
constant float3 white = float3(1.0, 1.0, 1.0);
constant float3 sky_color = float3(0.53, 0.81, 0.92);

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
  float transparency;
  float translucency;
  float refractive_index;
};

struct Sphere {
  packed_float3 center;
  float radius;
  Material material;
};

struct Triangle {
  packed_float3 v0;
  packed_float3 v1;
  packed_float3 v2;
  Material material;
};

struct Hit {
  float t;
  bool internal;
  packed_float3 normal;
  Material material;
};

constant Hit NO_HIT =
    Hit{INFINITY, false, packed_float3(0.0f, 0.0f, 0.0f),
        Material{packed_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f}};

float schlick_fresnel(float cosine, float ref_idx) {
  float r0 = (1 - ref_idx) / (1 + ref_idx);
  r0 = r0 * r0;
  return r0 + (1 - r0) * pow((1 - cosine), 5);
}

bool check_transmission(float transparency, float ref_idx, packed_float3 dir,
                        packed_float3 normal, thread SimpleRNG &rng) {
  if (transparency <= 0.0f) {
    return false;
  }
  float fresnel = schlick_fresnel(abs(dot(dir, normal)), ref_idx);
  float reflection_prob = fresnel * (1 - transparency);

  return rng.rand() > reflection_prob;
}

packed_float3 reflect_dir(packed_float3 dir, packed_float3 normal) {
  return dir - 2 * dot(dir, normal) * normal;
}

packed_float3 refract_dir(packed_float3 dir, packed_float3 normal,
                          float ref_idx) {
  float cos_theta = abs(dot(dir, normal));
  float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
  float sin_theta_prime = sin_theta / ref_idx;

  if (sin_theta_prime > 1.0) {
    packed_float3 reflected_dir = reflect_dir(dir, normal);
    return reflected_dir;
  }

  float cos_theta_prime = sqrt(1.0 - sin_theta_prime * sin_theta_prime);

  packed_float3 r_parallel = (dir + normal * cos_theta) / ref_idx;
  packed_float3 r_perp = -normal * cos_theta_prime;

  return normalize(r_parallel + r_perp);
}

packed_float3 perturb_dir(packed_float3 dir, float translucency,
                          thread SimpleRNG &rng) {
  float r1 = rng.rand_normal();
  float r2 = rng.rand_normal();
  float r3 = rng.rand_normal();
  return normalize(dir + translucency * packed_float3(r1, r2, r3));
}

bool check_specular(float reflectivity, thread SimpleRNG &rng) {
  return rng.rand() < reflectivity;
}

packed_float3 rand_dir(packed_float3 normal, thread SimpleRNG &rng) {
  float r1 = rng.rand_normal();
  float r2 = rng.rand_normal();
  float r3 = rng.rand_normal();
  return normalize(normal + packed_float3(r1, r2, r3));
}

packed_float3 diffuse_dir(packed_float3 normal, thread SimpleRNG &rng) {
  packed_float3 random_dir = rand_dir(normal, rng);
  if (dot(random_dir, normal) < 0) {
    random_dir = -random_dir;
  }
  return random_dir;
}

Ray add_environment(Ray ray) {
  packed_float3 color;
  if (ray.dir.z > .98) {
    float scale = (ray.dir.z - .98) / .02;
    color = scale * white + (1 - scale) * sun_color;
    ray.intensity += 1.0;
  } else {
    color = sky_color;
    ray.intensity += 0.5;
  }
  ray.color = ray.color * color;
  return ray;
}

Ray get_ray(constant View &view, uint base_id, thread SimpleRNG &rng) {
  float x_offset = static_cast<float>(base_id % view.width) + rng.rand() - 0.5f;
  int y_offset = static_cast<float>(base_id / view.width) + rng.rand() - 0.5f;

  return Ray{view.origin,
             normalize(view.top_left_dir + x_offset * view.right_dir +
                       y_offset * view.down_dir),
             packed_float3(1.0f, 1.0f, 1.0f), 0.0f};
}

Hit sphere_hit(Ray ray, Sphere sphere) {
  packed_float3 ray_offset_origin = ray.origin - sphere.center;
  float b = 2 * dot(ray.dir, ray_offset_origin);
  float c = length_squared(ray_offset_origin) - sphere.radius * sphere.radius;
  float discriminant = b * b - 4 * c;
  if (discriminant > 0) {
    float t = (-b - sqrt(discriminant)) / 2.0f;
    if (t > 0) {
      return Hit{
          t, false,
          normalize((ray.origin + t * ray.dir - sphere.center) / sphere.radius),
          sphere.material};
    }
  }
  return NO_HIT;
}

Hit _sphere_hit(Ray ray, Sphere sphere) {
  packed_float3 ray_offset_origin = ray.origin - sphere.center;
  float b = 2 * dot(ray.dir, ray_offset_origin);
  float c = length_squared(ray_offset_origin) - sphere.radius * sphere.radius;
  float discriminant = b * b - 4 * c;

  if (discriminant > 0) {
    float sqrt_d = sqrt(discriminant);
    float t1 = (-b - sqrt_d) / 2.0f;
    float t2 = (-b + sqrt_d) / 2.0f;

    float t;
    bool internal;
    if (t1 > epsilon) {
      t = t1;
      internal = false;
    } else if (t2 > epsilon) {
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

    return Hit{t, internal, normal, sphere.material};
  }
  return NO_HIT;
}

Hit triangle_hit(Ray ray, Triangle triangle) {
  // checks both faces
  float3 ab = triangle.v1 - triangle.v0;
  float3 ac = triangle.v2 - triangle.v0;
  float3 pvec = cross(ray.dir, ac);
  float det = dot(ab, pvec);

  if (abs(det) < epsilon) {
    return NO_HIT;
  }

  float inv_det = 1.0 / det;
  float3 tvec = ray.origin - triangle.v0;
  float u = dot(tvec, pvec) * inv_det;
  if (u < 0.0 || u > 1.0) {
    return NO_HIT;
  }

  float3 qvec = cross(tvec, ab);
  float v = dot(ray.dir, qvec) * inv_det;
  if (v < 0.0 || u + v > 1.0) {
    return NO_HIT;
  }

  float t = dot(ac, qvec) * inv_det;
  if (t < 10 * epsilon) {
    return NO_HIT;
  }

  float3 normal = normalize(cross(ab, ac));
  if (det < 0) {
    normal = -normal;
  }

  return Hit{t, false, normal, triangle.material};
}

kernel void trace_rays(constant View &view [[buffer(0)]],
                       const device Sphere *spheres [[buffer(1)]],
                       const device Triangle *triangles [[buffer(2)]],
                       constant int &num_spheres [[buffer(3)]],
                       constant int &num_triangles [[buffer(4)]],
                       constant int &num_bounces [[buffer(5)]],
                       constant int &num_rays [[buffer(6)]],
                       constant int &seed [[buffer(7)]],
                       constant float &exposure [[buffer(8)]],
                       device packed_float3 *image [[buffer(9)]],
                       uint id [[thread_position_in_grid]]) {
  SimpleRNG rng = SimpleRNG(seed, id * id);

  packed_float3 pixel = packed_float3(0.0f, 0.0f, 0.0f);

  int last_triangle_hit = -1;
  int current_triangle_hit = -1;

  for (int ray_num = 0; ray_num < num_rays; ray_num++) {
    Ray ray = get_ray(view, id, rng);

    for (int bounce = 0; bounce < num_bounces; bounce++) {
      Hit closestHit = NO_HIT;

      for (int sphere_id = 0; sphere_id < num_spheres; sphere_id++) {
        Hit hit = sphere_hit(ray, spheres[sphere_id]);
        if (hit.t < closestHit.t) {
          closestHit = hit;
        }
      }

      for (int triangle_id = 0; triangle_id < num_triangles; triangle_id++) {
        if (triangle_id == last_triangle_hit) {
          last_triangle_hit = -1;
          continue;
        }
        Hit hit = triangle_hit(ray, triangles[triangle_id]);
        if (hit.t < closestHit.t) {
          current_triangle_hit = triangle_id;
          closestHit = hit;
        }
      }

      if (closestHit.t < INFINITY) {
        if (current_triangle_hit != -1) {
          last_triangle_hit = current_triangle_hit;
          current_triangle_hit = -1;
        }
        ray.origin = ray.origin + closestHit.t * ray.dir;
        bool is_specular =
            check_specular(closestHit.material.reflectivity, rng);
        if (is_specular) {
          ray.dir = reflect_dir(ray.dir, closestHit.normal);
          ray.origin = ray.origin + epsilon * closestHit.normal;
        } else {
          ray.color = ray.color * closestHit.material.color;
          ray.intensity += closestHit.material.intensity;
          ray.dir = diffuse_dir(closestHit.normal, rng);
          ray.dir = perturb_dir(ray.dir, closestHit.material.translucency, rng);
        }
      } else {
        ray = add_environment(ray);
        break;
      }
    }

    pixel += ray.color * ray.intensity;
  }

  // tone map
  image[id] = ((1 - exp(-pixel * exposure / num_rays)) * 255.0);
}
