import "common";

[shader("closesthit")]
void main(inout RayPayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    MeshInfo meshInfo = meshInfoBuffer[InstanceID()];

    uint primitive = PrimitiveIndex();
    uint i0 = meshInfo.indices[primitive * 3 + 0];
    uint i1 = meshInfo.indices[primitive * 3 + 1];
    uint i2 = meshInfo.indices[primitive * 3 + 2];

    float3 n0 = meshInfo.vertices[i0].normal;
    float3 n1 = meshInfo.vertices[i1].normal;
    float3 n2 = meshInfo.vertices[i2].normal;

    float3 bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics);
    float3 normal = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);
    payload.color = normal * 0.5 + 0.5;
}
