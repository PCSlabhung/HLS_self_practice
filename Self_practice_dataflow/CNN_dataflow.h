#ifndef _CNN_dataflow_H
#define _CNN_dataflow_H
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include <fstream>
#include <iostream>
#include <string>
using namespace std;
typedef ap_int<32> int_32;
typedef ap_fixed<10,6,AP_RND_CONV, AP_SAT> fixed_16;
void top_model(fixed_16 input_imag[32][32], fixed_16 out_result[10]);
// void load_weight(fixed_16 input_filter[3][3], string input_file){
//     ifstream in_weight;
//     in_weight.open(input_file);
//     if(!in_weight){
//         cout<<"fail to open";
//     }
//     else{
//         for(int i = 0; i < 3;i++){
//             for(int j = 0; j < 3; j++){
//                 in_weight >> input_filter[i][j];
//             }
//         }
//     }
// }
template<int in_size, int out_size>
void load_weight_FC(fixed_16 filter[out_size][in_size], string input_file){
    ifstream in_weight;
    in_weight.open(input_file);
    if(!in_weight){
        cout<<"fail to open"<<"\n";
    }
    else{
        for(int i = 0; i < out_size; i++){
            for(int j = 0; j < in_size; j++){
                in_weight >> filter[i][j];
            }
        }
    }
}
template<int in_size, int out_size>
void FC(fixed_16 input[in_size], fixed_16 filter[out_size][in_size],fixed_16 output[out_size]){
    for(int i = 0; i < out_size; i++){
        for(int j = 0 ;j < in_size; j++){
            output[i] = input[j] * filter[i][j];
        }
    }
} 
#endif 