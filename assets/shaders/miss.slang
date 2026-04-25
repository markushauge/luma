import "common";

[shader("miss")]
void main(inout RayPayload payload)
{
    payload.color = float3(0.1);
}
