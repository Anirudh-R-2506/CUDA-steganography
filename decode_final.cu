#include <bits/stdc++.h>
#include <highgui.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace std;
using namespace cv;

string BinaryStringToText(string binaryString) {
  string text = "";
  stringstream sstream(binaryString);
  while (sstream.good()) {
    bitset<8> bits;
    sstream >> bits;
    text += char(bits.to_ulong());
  }
  return text;
}

__global__ void LSB(unsigned char *input, char *message, int image_size) {
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  int offset = xIndex + yIndex * blockDim.x * gridDim.x;

  if (offset >= image_size) {
    return;
  }

  message[offset] = 0;
  if (input[offset] & 1) {
    message[offset] |= 1;
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    cout << "Arguments Error"
         << "\n";
    exit(-1);
  }

  Mat image = imread(argv[1]);
  if (image.empty()) {
    cout << "Image Error\n";
    exit(-1);
  }
  const int imageByte = image.step * image.rows;
  int image_size = image.cols * image.rows;
  unsigned char *d_input;
  char *message_d, *message_h;

  message_h = (char *)malloc(imageByte * sizeof(char));
  cudaMalloc((void **)&message_d, imageByte * sizeof(char));
  cudaMalloc<unsigned char>(&d_input, imageByte);

  cudaMemcpy(d_input, image.ptr(), imageByte, cudaMemcpyHostToDevice);

  const dim3 block(16, 16);
  const dim3 grid((image.cols + block.x - 1) / block.x,
                  (image.rows + block.y - 1) / block.y);

  // capture the start time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  LSB<<<grid, block>>>(d_input, message_d, image_size);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Decode Kernel execution time is:  %3.10f sec\n", elapsedTime / 1000);

  cudaMemcpy(message_h, message_d, imageByte * sizeof(char),
             cudaMemcpyDeviceToHost);
  int i = 0, j = 0;
  while (i < imageByte - 8) {
    string oneChar = "";
    for (j = 0; j < 8; j++) {
      int index = i + j;
      int num = (int)message_h[index];
      char temp[1];
      sprintf(temp, "%d", num);
      string s(temp);
      oneChar += s;
    }

    if (oneChar == "00000000") {
      break;
    }

    String ch = BinaryStringToText(oneChar);
    cout << ch;
    i += 8;
  }
  cout << "\n";
  return 0;
}
