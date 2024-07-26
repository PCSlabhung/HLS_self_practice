#include "CNN_dataflow.h"
#include "hls_stream.h"
#include "FC_weight.h"

#define input_size 32

template<int in_size>
void flatten(fixed_16 input[in_size][in_size], fixed_16 out[in_size * in_size]){
    for(int i = 0; i < in_size; i++){
        for(int j = 0; j < in_size; j++){
            out[i * 4 + j] = input[i][j];
        }
    }
}

template<int in_size>
void maxpool(fixed_16 input[in_size][in_size], fixed_16 output[in_size / 2][in_size / 2]){

	for(int i = 0; i < in_size / 2; i ++){
        for(int j = 0; j < in_size / 2; j ++){

            int index_i = i * 2;
            int index_j = j * 2;
            if(input[index_i][index_j] > input[index_i][index_j + 1] && input[index_i][index_j] > input[index_i + 1][index_j] && input[index_i][index_j] > input[index_i + 1][index_j + 1]){
                output[i][j] = input[index_i][index_j];
            }
            else if(input[index_i][index_j + 1] > input[index_i][index_j] && input[index_i][index_j + 1] > input[index_i + 1][index_j] && input[index_i][index_j + 1] > input[index_i + 1][index_j + 1]){
                output[i][j] = input[index_i][index_j + 1];
            }
            else if(input[index_i + 1][index_j] > input[index_i][index_j] && input[index_i + 1][index_j] > input[index_i][index_j + 1] && input[index_i + 1][index_j] > input[index_i + 1][index_j + 1]){
                output[i][j] = input[index_i + 1][index_j];
            }
            else{
                output[i][j] = input[index_i + 1][index_j + 1];
            }
        }
    }
}
template<int num>
void ReLU(fixed_16 input[num][num], fixed_16 output[num][num]){

	for(int i = 0; i < num; i++){
        for(int j = 0; j < num; j++){

            if(input[i][j] < 0){
                output[i][j] = 0;
            }
            else{
                output[i][j] = input[i][j];
            }
        }
    }

}
template <int in_size>
void padding(fixed_16 input_imag[in_size][in_size], fixed_16 out_imag[in_size + 2][in_size + 2]){
    for(int i = 0; i <= in_size + 1; i++){
        out_imag[0][i] = 0;
        out_imag[i][0] = 0;
        out_imag[in_size + 1][i] = 0;
        out_imag[i][in_size + 1] = 0;
    }
    for(int i = 1; i <= in_size; i++){
        for(int j = 1; j <= in_size;j++){
            out_imag[i][j] = input_imag[i - 1][j - 1];
        }
    }
    
}
template<int in_size, int out_size>
void CNN(fixed_16 input_imag[in_size][in_size], fixed_16 input_filter[3][3], fixed_16 output_imag[out_size][out_size]){
    fixed_16 padding_imag[in_size + 2][in_size + 2];
    
    padding<in_size>(input_imag, padding_imag);

    for(int i = 0; i < in_size; i++){
#pragma HLS PIPELINE
        for(int j = 0; j < in_size; j++){
            

            output_imag[i][j] = (padding_imag[i][j] * input_filter[0][0])+(padding_imag[i][j + 1] * input_filter[0][1])+(padding_imag[i][j + 2] * input_filter[0][2])
                            +(padding_imag[i + 1][j] * input_filter[1][0])+(padding_imag[i + 1][j + 1] * input_filter[1][1])+(padding_imag[i + 1][j + 2] * input_filter[1][2])
                            +(padding_imag[i + 2][j] * input_filter[2][0])+(padding_imag[i + 2][j + 1] * input_filter[2][1])+(padding_imag[i + 2][j + 2] * input_filter[2][2]); 
        }
    }
}
void top_model(fixed_16 input_imag[input_size][input_size], fixed_16 output[10]){
    fixed_16 input_filter[3][3][3] = {{{0.788226, 0.823445, -0.363334 },{-0.642831, -0.0363487, -0.906763},{-0.901575, -0.823079, 0.830159}}
    ,{{0.692395, 0.178813, 0.885705 },{-0.891076, 0.879235, 0.833211},{0.422359, 0.566166, 0.607123}},
    {{0.449033, -0.116432, -0.864707 },{0.403925, -0.544986, 0.944851 },{-0.0953122, -0.33727, 0.70094 }}};
 //   fixed_16 fc[10][16];
	#pragma HLS DATAFLOW
    fixed_16 layer1_CNN_out[input_size][input_size];
    fixed_16 layer1_max_out[input_size/2][input_size/2];
    fixed_16 layer1_out[input_size/2][input_size/2];
#pragma HLS STREAM variable= layer1_CNN_out type=fifo
#pragma HLS STREAM variable= layer1_max_out type=fifo
#pragma HLS STREAM variable= layer1_out type=fifo
    #ifndef __SYNTHESIS__
        for(int i = 0; i < 10 ; i++){
            for(int j = 0; j < 16 ; j++){
                cout<<fc[i][j]<<" ";
            }
            cout<<"\n";
        }
    #endif
    
    CNN<input_size,input_size>(input_imag,input_filter[0], layer1_CNN_out);
    maxpool<input_size>(layer1_CNN_out, layer1_max_out);
    ReLU<input_size/2>(layer1_max_out, layer1_out);
    
    fixed_16 layer2_CNN_out[input_size/2][input_size/2];
    fixed_16 layer2_max_out[input_size/4][input_size/4];
    fixed_16 layer2_out[input_size/4][input_size/4];
    CNN<input_size/2, input_size/2>(layer1_out, input_filter[1], layer2_CNN_out);
    maxpool<input_size/2>(layer2_CNN_out, layer2_max_out);
    ReLU<input_size/4>(layer2_max_out, layer2_out);
    
    fixed_16 layer3_CNN_out[input_size/4][input_size/4];
    fixed_16 layer3_max_out[input_size/8][input_size/8];
    fixed_16 layer3_out[input_size/8][input_size/8];
    CNN<input_size/4, input_size/4>(layer2_out, input_filter[2], layer3_CNN_out);
    maxpool<input_size/4>(layer3_CNN_out, layer3_max_out);
    ReLU<input_size/8>(layer3_max_out, layer3_out);
    
    fixed_16 flattened[16];
    flatten<4>(layer3_out, flattened);
    
    FC<16, 10>(flattened, fc, output);

}
