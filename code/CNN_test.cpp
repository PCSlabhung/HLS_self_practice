#include "CNN_test.h"
#include <iostream>
using namespace std;
#define pararell_factor 3
void CNN(int_32 filter[3][3], int_32 image[9][9], int_32 output[7][7]){
	#pragma HLS INTERFACE ap_none port=filter
	#pragma HLS INTERFACE ap_none port = image
	#pragma HLS INTERFACE ap_none port = output


	#pragma HLS ARRAY_PARTITION variable = filter type = cyclic factor = pararell_factor dim=2
	#pragma HLS ARRAY_PARTITION variable = image  type = cyclic factor = pararell_factor dim=2

	//CNN
#pragma HLS PIPELINE II = 1 rewind
	for(int i = 0; i < 7; i++){

		for(int j = 0 ; j < 7; j++){
			#pragma HLS unroll factor = pararell_factor * 2
			output[i][j] = image[i][j] * filter[0][0] + image[i][j + 1] * filter[0][1] + image[i][j + 2] * filter[0][2] +
					image[i + 1][j] * filter[1][0] + image[i + 1][j + 1] * filter[1][1] + image[i + 1][j + 2] * filter[1][2] +
					image[i + 2][j] * filter[2][0] + image[i + 2][j + 1] * filter[2][1] + image[i + 2][j + 2] * filter[2][2];
		}
	}
	return;
}

