// Bindings

[[vk::binding(0)]]
RWTexture2D<float4> image;

[[vk::binding(1)]]
RaytracingAccelerationStructure topLevelAS;

struct Vertex
{
    float3 position;
    float3 normal;
    float2 uv;
}

struct MeshInfo
{
    Vertex *vertices;
    uint *indices;
}

[[vk::binding(2)]]
StructuredBuffer<MeshInfo> meshInfoBuffer;

struct PushConstants
{
    float3 cameraTranslation;
    float3x3 cameraRotation;
    float cameraFov;
}

[[vk::push_constant]]
PushConstants pc;

// Ray tracing pipeline

struct RayPayload
{
    float3 color;
}
