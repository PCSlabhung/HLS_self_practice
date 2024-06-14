#ifndef _CNN_H
#define _CNN_H
#include "ap_int.h"
typedef ap_int<32> int_32;
void CNN(int_32 input_filter[3][3], int_32 input_image[9][9], int_32 output[7][7]);
#endif
