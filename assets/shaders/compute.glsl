#version 460

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba8) uniform writeonly image2D img;

layout(push_constant) uniform PushConstants {
    int width;
    int height;
    int frameCount;
} pc;

const float EPSILON = 0.0005;
const int MAX_STEPS = 1000;
const float MAX_DISTANCE = 1000.0;
const vec3 SKY_COLOR = vec3(0.6, 0.8, 1.0);
const vec3 AMBIENT_LIGHT = vec3(0.1);

struct HitInfo {
    float t;
    vec3 position;
    vec3 normal;
};

float sdSphere(vec3 p, float s) {
    return length(p) - s;
}

float scene(vec3 p) {
    return sdSphere(p, 1.0);
}

vec3 sceneNormal(vec3 p) {
    vec2 e = vec2(EPSILON, 0.0);
    float dx = scene(p + e.xyy) - scene(p - e.xyy);
    float dy = scene(p + e.yxy) - scene(p - e.yxy);
    float dz = scene(p + e.yyx) - scene(p - e.yyx);
    return normalize(vec3(dx, dy, dz));
}

HitInfo raymarch(vec3 ro, vec3 rd) {
    HitInfo hit;
    float t = 0.0;

    for (int i = 0; i < MAX_STEPS; ++i) {
        vec3 p = ro + rd * t;
        float d = scene(p);

        if (d < EPSILON) {
            hit.t = t;
            hit.position = p;
            hit.normal = sceneNormal(p);
            return hit;
        }

        t += d;

        if (t > MAX_DISTANCE) {
            break;
        }
    }

    hit.t = -1.0;
    return hit;
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    if (coord.x >= pc.width || coord.y >= pc.height) {
        return;
    }

    vec2 uv = vec2(
        float(coord.x) / float(pc.width - 1),
        1.0 - float(coord.y) / float(pc.height - 1)
    );

    vec2 ndc = uv * 2.0 - 1.0;
    ndc.x *= float(pc.width) / float(pc.height);

    vec3 rayOrigin = vec3(0.0, 0.0, -2.0);
    vec3 rayDirection = normalize(vec3(ndc, 1.0));

    HitInfo hit = raymarch(rayOrigin, rayDirection);
    vec3 color = SKY_COLOR;

    vec3 lightDir = normalize(vec3(
        sin(pc.frameCount * 0.01),
        1.0,
        cos(pc.frameCount * 0.01)
    ));

    if (hit.t >= 0.0) {
        vec3 diffuse = max(dot(hit.normal, lightDir), 0.0) * vec3(1.0);
        color = diffuse * 0.9 + AMBIENT_LIGHT;
    }

    imageStore(img, coord, vec4(color, 1.0));
}
