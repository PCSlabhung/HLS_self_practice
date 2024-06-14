#include <iostream>
#include "CNN_test.h"
using namespace std;
int main(){
	cout <<" >>>>>>>>>>  start test  <<<<<<<<<<\n";
	cout<<">>>>>>>>  input filter  <<<<<<<<\n";
	int_32 filter[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
	int_32 image[9][9];
	for(int i = 0; i < 9; i ++){
		for(int j = 0 ; j < 9 ; j++){
			image[i][j] = (i + j) % 5;
		}
	}
	int_32 out[7][7];
	CNN(filter, image, out);
	for(int i = 0 ; i < 7 ; i++){
		for(int j = 0 ; j < 7; j++){
				cout<<out[i][j]<<" ";
		}
		cout<<endl;
	}
	for(int i = 0; i < 7; i++){
		for(int j = 0; j < 7; j++){
			int golden = image[i][j] * filter[0][0] + image[i][j + 1] * filter[0][1] + image[i][j + 2] * filter[0][2] +
					image[i + 1][j] * filter[1][0] + image[i + 1][j + 1] * filter[1][1] + image[i + 1][j + 2] * filter[1][2] +
					image[i + 2][j] * filter[2][0] + image[i + 2][j + 1] * filter[2][1] + image[i + 2][j + 2] * filter[2][2];
			if(out[i][j] != golden){
				cout<<"i = "<<i <<" j = "<<j <<"out = "<<out[i][j] << "golden = "<<golden;
				return 1;
			}
		}
	}
	return 0;
}
