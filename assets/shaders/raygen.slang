struct RayPayload
{
    float3 color;
}

[[vk::binding(0)]]
RWTexture2D<float4> image;

[[vk::binding(1)]]
RaytracingAccelerationStructure topLevelAS;

struct PushConstants
{
    uint viewportWidth;
    uint viewportHeight;
    float3 cameraTranslation;
    float3x3 cameraRotation;
    float cameraFov;
    uint timeMillis;
}

[[vk::push_constant]]
PushConstants pc;

[shader("raygeneration")]
void main()
{
    uint2 coord = DispatchRaysIndex().xy;

    if (coord.x >= pc.viewportWidth || coord.y >= pc.viewportHeight)
    {
        return;
    }

    float2 uv = float2(
        float(coord.x) / float(pc.viewportWidth - 1),
        1.0 - float(coord.y) / float(pc.viewportHeight - 1));

    float2 ndc = uv * 2.0 - 1.0;
    float aspectRatio = float(pc.viewportWidth) / float(pc.viewportHeight);
    float tanHalfFov = tan(pc.cameraFov * 0.5);
    float3 rayDir = float3(ndc.x * aspectRatio * tanHalfFov, ndc.y * tanHalfFov, -1.0);
    rayDir = normalize(mul(pc.cameraRotation, rayDir));

    RayDesc ray;
    ray.Origin = pc.cameraTranslation;
    ray.Direction = rayDir;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    RayPayload payload;
    TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, ray, payload);

    image[int2(coord)] = float4(payload.color, 1.0);
}
