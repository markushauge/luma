struct Camera {
    vec3 position;
    mat3 rotation;
    float fov;
};

mat3 cameraLookAt(vec3 eye, vec3 target, vec3 up) {
    vec3 f = normalize(target - eye);
    vec3 r = normalize(cross(f, up));
    vec3 u = cross(r, f);
    return mat3(r, u, f);
}

// Converts a full frame (36x24mm) focal length to a vertical field of view in radians
float focalLengthToVerticalFov(float focalLength) {
    return 2.0 * atan(24.0 / (2.0 * focalLength));
}
