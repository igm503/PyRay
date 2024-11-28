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
  thread SimpleRNG(const unsigned seed1, const unsigned seed2 = 1);
  thread float rand();
  thread float rand_normal();
};

thread SimpleRNG::SimpleRNG(const uint seed1, const uint seed2) {
  this->state = seed1 * 1103515245 + seed2 * 4928004 / seed1;
}

thread float SimpleRNG::rand() {
  this->state = (this->state + 51) * 1103515245 + 12345;
  return float(this->state & 0x7FFFFFFF) / float(0x7FFFFFFF);
}
thread float SimpleRNG::rand_normal() {
  // mean is 0; standard deviation is 1
  return sqrt(-2.0 * log(this->rand())) * sin(2.0 * M_PI_F * this->rand());
}

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
  bool hit;
  float t;
  packed_float3 normal;
  Material material;
};

constant Hit no_hit =
    Hit{false, 0.0f, packed_float3(0.0f, 0.0f, 0.0f),
        Material{packed_float3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f}};

bool check_specular(float reflectivity, thread SimpleRNG &rng) {
  return rng.rand() < reflectivity;
}

packed_float3 reflect_dir(packed_float3 dir, packed_float3 normal) {
  return dir - 2 * dot(dir, normal) * normal;
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

Hit sphere_hit(Ray ray, Sphere sphere) {
  packed_float3 ray_offset_origin = ray.origin - sphere.center;
  float b = 2 * dot(ray.dir, ray_offset_origin);
  float c =
      dot(ray_offset_origin, ray_offset_origin) - sphere.radius * sphere.radius;
  float discriminant = b * b - 4 * c;
  if (discriminant > 0) {
    float t = (-b - sqrt(discriminant)) / 2.0f;
    if (t > 0) {
      return Hit{
          true, t,
          normalize((ray.origin + t * ray.dir - sphere.center) / sphere.radius),
          sphere.material};
    }
  }
  return no_hit;
}

Hit triangle_hit(Ray ray, Triangle triangle) {
  float3 ab = triangle.v1 - triangle.v0;
  float3 ac = triangle.v2 - triangle.v0;
  float3 pvec = cross(ray.dir, ac);
  float det = dot(ab, pvec);

  if (det < epsilon) {
    return no_hit;
  }

  float inv_det = 1.0 / (det + epsilon);
  float3 tvec = ray.origin - triangle.v0;
  float u = dot(tvec, pvec) * inv_det;
  if (u < 0.0 || u > 1.0) {
    return no_hit;
  }

  float3 qvec = cross(tvec, ab);
  float v = dot(ray.dir, qvec) * inv_det;
  if (v < 0.0 || u + v > 1.0) {
    return no_hit;
  }

  float t = dot(ac, qvec) * inv_det;
  if (t < 10 * epsilon) {
    return no_hit;
  }
  return Hit{true, t, normalize(cross(ab, ac)), triangle.material};
}

Ray get_ray(constant View &view, uint base_id, thread SimpleRNG &rng) {
  float x_offset = static_cast<float>(base_id % view.width) + rng.rand() - 0.5f;
  int y_offset = static_cast<float>(base_id / view.width) + rng.rand() - 0.5f;

  return Ray{view.origin,
             normalize(view.top_left_dir + x_offset * view.right_dir +
                       y_offset * view.down_dir),
             packed_float3(1.0f, 1.0f, 1.0f), 0.0f};
}

kernel void trace_rays(constant View &view [[buffer(0)]],
                       const device Sphere *spheres [[buffer(1)]],
                       const device Triangle *triangles [[buffer(2)]],
                       constant int &num_spheres [[buffer(3)]],
                       constant int &num_triangles [[buffer(4)]],
                       constant int &num_bounces [[buffer(5)]],
                       constant int &num_rays [[buffer(6)]],
                       constant int &seed [[buffer(7)]],
                       device packed_float3 *image [[buffer(8)]],
                       uint id [[thread_position_in_grid]]) {
  SimpleRNG rng = SimpleRNG(seed * 400, id * id);

  for (int ray_num = 0; ray_num < num_rays; ray_num++) {
    Ray ray = get_ray(view, id, rng);

    for (int bounce = 0; bounce < num_bounces; bounce++) {
      Hit closestHit;
      closestHit.hit = false;
      closestHit.t = INFINITY;
      for (int sphere_id = 0; sphere_id < num_spheres; sphere_id++) {
        Sphere sphere = spheres[sphere_id];
        Hit hit = sphere_hit(ray, sphere);
        if (hit.hit && hit.t < closestHit.t) {
          closestHit = hit;
        }
      }
      for (int triangle_id = 0; triangle_id < num_triangles; triangle_id++) {
        Triangle triangle = triangles[triangle_id];
        Hit hit = triangle_hit(ray, triangle);
        if (hit.hit && hit.t < closestHit.t) {
          closestHit = hit;
        }
      }
      if (closestHit.hit) {
        ray.origin = ray.origin + closestHit.t * ray.dir;
        bool is_specular =
            check_specular(closestHit.material.reflectivity, rng);
        if (is_specular) {
          ray.dir = reflect_dir(ray.dir, closestHit.normal);
        } else {
          ray.color = ray.color * closestHit.material.color;
          ray.intensity += closestHit.material.intensity;
          ray.dir = diffuse_dir(closestHit.normal, rng);
        }
      } else {
        ray = add_environment(ray);
        break;
      }
    }

    image[id] += ray.color * ray.intensity;
  }
  image[id] /= num_rays;
}
kernel void tone_map(const device packed_float3 *hdr_image [[buffer(0)]],
                     constant float &exposure [[buffer(1)]],
                     device packed_float3 *ldr_image [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
  ldr_image[id] = ((1 - exp(-hdr_image[id] * exposure)) * 255.0);
}
