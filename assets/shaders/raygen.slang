import "common";

[shader("raygeneration")]
void main()
{
    const uint2 index = DispatchRaysIndex().xy;
    const uint2 dimensions = DispatchRaysDimensions().xy;
    float2 uv = (float2)index / (float2)dimensions;
    uv.y = 1.0 - uv.y;

    float2 ndc = uv * 2.0 - 1.0;
    float aspectRatio = (float)dimensions.x / (float)dimensions.y;
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

    image[index] = float4(payload.color, 1.0);
}
