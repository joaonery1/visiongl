/** Convert from RGB to grayscale.

  */

__kernel void vglClRgb2Gray(__read_only image2d_t img_input,__write_only image2d_t img_output)
{
  int2 coords = (int2)(get_global_id(0), get_global_id(1));
  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
                        CLK_ADDRESS_CLAMP |           //Clamp to zeros
                        CLK_FILTER_NEAREST;           //Don't interpolate

  float4 p = read_imagef(img_input, smp, coords);

  float4 o;
  float val = 0.299 * p.x + 0.587 * p.y + 0.114 * p.z;
  o.x = val;
  o.y = val;
  o.z = val;
  o.w = 1.0;

  //o = val;
  write_imagef(img_output, coords, o);
}
