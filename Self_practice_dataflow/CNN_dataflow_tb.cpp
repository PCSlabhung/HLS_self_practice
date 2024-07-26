#include "CNN_dataflow.h"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
using namespace std;
void load_filter(fixed_16 filter[3][3], ifstream& infile){
    for(int i = 0;i < 3;i++){
        for(int j = 0; j < 3; j++){
            infile >> filter[i][j];
        }
    }
}
template<int size>
void fc(fixed_16 input[size], fixed_16 filter[size], fixed_16 &out){
    fixed_16 sum = 0;
    for(int i = 0; i < size; i++){
        sum += input[i] * filter[i];
    }
    out = sum;
}
template <int size>
void layer(fixed_16 input_img[size][size], fixed_16 filter[3][3],fixed_16 output[size/2][size/2]){
    fixed_16 layer_CNN[size][size];
    for(int i = 0; i < size;i++){
        for(int j = 0; j < size; j++){
            if(i == 0 && j == 0){
                layer_CNN[i][j] = input_img[0][0] * filter[1][1] + input_img[0][1] * filter[1][2] + 
                                    input_img[1][0] * filter[2][1] + input_img[1][1] * filter[2][2];
            }
            else if(i == 0 && j == size - 1){
                layer_CNN[i][j] = input_img[0][size - 1] * filter[1][1] + input_img[0][size - 2] * filter[1][0] + 
                                    input_img[1][size - 1] * filter[2][1] + input_img[1][size - 2] * filter[2][0];
            }
            else if(i == size - 1 && j == size - 1){
                layer_CNN[i][j] = input_img[size - 1][size - 1] * filter[1][1] + input_img[size - 1][size - 2] * filter[1][0] + 
                                    input_img[size - 2][size - 1] * filter[0][1] + input_img[size - 2][size - 2] * filter[0][0];
            }
            else if(i == size - 1 && j == 0){
                layer_CNN[i][j] = input_img[size - 1][0] * filter[1][1] + input_img[size - 1][1] * filter[1][2] + 
                                    input_img[size - 2][0] * filter[0][1] + input_img[size - 2][1] * filter[0][2];
            }
            else if(i == 0){
                layer_CNN[i][j] = input_img[i][j - 1] * filter[1][0] + input_img[i][j] * filter[1][1] + input_img[i][j + 1] * filter[1][2] + 
                                    input_img[i + 1][j - 1] * filter[2][0] + input_img[i + 1][j] * filter[2][1] + input_img[i + 1][j + 1] * filter[2][2];
            }
            else if(j == 0){
                layer_CNN[i][j] = input_img[i - 1][j] * filter[0][1] + input_img[i - 1][j + 1] * filter[0][2] + 
                                    input_img[i][j] * filter[1][1] + input_img[i][j + 1] * filter[1][2] + 
                                    input_img[i + 1][j] * filter[2][1] + input_img[i + 1][j + 1] * filter[2][2];
            }
            else if(i == size - 1){
                layer_CNN[i][j] = input_img[i - 1][j - 1] * filter[0][0] + input_img[i - 1][j] * filter[0][1] + input_img[i - 1][j + 1] * filter[0][2] + 
                                    input_img[i][j - 1] * filter[1][0] + input_img[i][j] * filter[1][1] + input_img[i][j + 1] * filter[1][2];
            }
            else if(j == size - 1){
                layer_CNN[i][j] = input_img[i - 1][j - 1] * filter[0][0] + input_img[i - 1][j] * filter[0][1] + 
                                    input_img[i][j - 1] * filter[1][0] + input_img[i][j] * filter[1][1] + 
                                    input_img[i + 1][j - 1] * filter[2][0] + input_img[i + 1][j] * filter[2][1]; 
            }
            else {
                layer_CNN[i][j] = input_img[i - 1][j - 1] * filter[0][0] + input_img[i - 1][j] * filter[0][1] + input_img[i - 1][j + 1] * filter[0][2] + 
                                    input_img[i][j - 1] * filter[1][0] + input_img[i][j] * filter[1][1] + input_img[i][j + 1] * filter[1][2] + 
                                    input_img[i + 1][j - 1] * filter[2][0] + input_img[i + 1][j] * filter[2][1] + input_img[i + 1][j + 1] * filter[2][2]; 
            }
        }
    }
  
        
    for(int i = 0; i < size; i += 2){
        for(int j = 0; j < size; j += 2){
            if(layer_CNN[i][j] > layer_CNN[i][j + 1] && layer_CNN[i][j] > layer_CNN[i + 1][j] && layer_CNN[i][j] > layer_CNN[i + 1][j + 1]){
                output[i / 2][j / 2] = layer_CNN[i][j];
            }
            else if(layer_CNN[i + 1][j] > layer_CNN[i][j] && layer_CNN[i + 1][j] > layer_CNN[i][j + 1] && layer_CNN[i + 1][j] > layer_CNN[i + 1][j + 1]){
                output[i / 2][j / 2] = layer_CNN[i + 1][j];
            }
            else if(layer_CNN[i][j + 1] > layer_CNN[i][j] && layer_CNN[i][j + 1] > layer_CNN[i + 1][j] && layer_CNN[i][j + 1] > layer_CNN[i + 1][j + 1]){
                output[i / 2][j / 2] = layer_CNN[i][j + 1];
            }
            else{
                output[i / 2][j / 2] = layer_CNN[i + 1][j + 1];
            }
        }
    }
    
    for(int i = 0; i < size / 2; i++){
        for(int j = 0; j < size / 2;j++){
            if(output[i][j] < 0){
                output[i][j] = 0;
            }
        }
    }
}
int main(){
    fixed_16 test_imag[32][32];
    fixed_16 outcome[10];
    fixed_16 golden[10];
    //srand(time(NULL));
    srand(0);
    for(int i = 0; i < 32; i++){
        for(int j = 0; j < 32; j++){
            test_imag[i][j] = ((float)rand() - (float)(RAND_MAX / 2)) / (float)(RAND_MAX / 2);
        }
    }
    
    //test

    top_model(test_imag, outcome);
    //

    fixed_16 filter[3][3][3];
    fixed_16 fc_filter[10][16];
    load_weight_FC<16, 10>(fc_filter, "/home/hung52852/self_practice_CNN_dataflow/weight_FC.txt");
    ifstream infile;
    infile.open("/home/hung52852/self_practice_CNN_dataflow/weight_0.txt");
    load_filter(filter[0], infile);
    infile.close();
    infile.open("/home/hung52852/self_practice_CNN_dataflow/weight_1.txt");
    load_filter(filter[1], infile);
    infile.close();
    infile.open("/home/hung52852/self_practice_CNN_dataflow/weight_2.txt");
    load_filter(filter[2], infile);
    infile.close();
    
    // first layer
    fixed_16 first_layer[16][16];
    layer<32>(test_imag, filter[0], first_layer);

    fixed_16 second_layer[8][8];
    layer<16>(first_layer, filter[1], second_layer);
    

    fixed_16 third_layer[4][4];
    layer<8>(second_layer, filter[2], third_layer);
    

    fixed_16 flatten_layer[16];
    for(int i = 0; i < 16; i++){
        flatten_layer[i] = third_layer[i / 4][i % 4];
    }
    
    FC<16, 10>(flatten_layer, fc_filter, golden);
    for(int i = 0; i < 10; i++){
        if(outcome[i] != golden[i]){
            for(int j = 0; j < 10; j++){
                cout<<j<<"th outcome = "<<outcome[j]<<" \n";
            }
            cout<<"\n";
            for(int j = 0; j < 10; j++){
                cout<<j<<"th result = "<<golden[j]<<" \n";
            }
            return 1;
        }
    }
    return 0;
}