struct RayPayload
{
    float3 color;
}

[shader("closesthit")]
void main(inout RayPayload payload)
{
    payload.color = float3(0.5);
}
