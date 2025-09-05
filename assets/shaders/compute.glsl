#version 460

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba8) uniform writeonly image2D img;

layout(push_constant) uniform PushConstants {
    int width;
    int height;
} pc;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    vec2 uv = vec2(
        float(coord.x) / float(pc.width - 1),
        1.0 - float(coord.y) / float(pc.height - 1)
    );

    vec4 color = vec4(uv, 0.0, 1.0);
    imageStore(img, coord, color);
}
