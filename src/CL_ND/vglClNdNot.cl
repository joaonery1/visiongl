/** Invert N-dimensional image.

  */

__kernel void vglClNdNot(__global char* img_input, __global char* img_output)
{

  int coord = (  (get_global_id(2) - get_global_offset(2)) * get_global_size(1) * get_global_size(0)) +
              (  (get_global_id(1) - get_global_offset(1)) * get_global_size (0)  ) +
                 (get_global_id(0) - get_global_offset(0));

  img_output[coord] = 255 - img_input[coord];

}
