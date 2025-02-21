__kernel void vglClEqual(__read_only image2d_t img_input1,__read_only image2d_t img_input2, __global bool* output)
{
    if(output[0] == 1) return;

    int2 coords = (int2)(get_global_id(0), get_global_id(1));
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
                          CLK_ADDRESS_CLAMP |           //Clamp to zeros
                          CLK_FILTER_NEAREST;           //Don't interpolate
    float4 p1 = read_imagef(img_input1, smp, coords);
    float4 p2 = read_imagef(img_input2, smp, coords);
    if (!(p1.x == p2.x))
    {
        //printf("x NEQ! @ (%d, %d): p1 = %f != %f = p2\n", coords.x, coords.y, p1.x, p2.x);
        output[0] = 1;
    }
    if (!(p1.y == p2.y))
    {
        output[0] = 1;
    }
    if (!(p1.z == p2.z))
    {
        output[0] = 1;
    }
    if (!(p1.w == p2.w))
    {
        output[0] = 1;
    }
}
