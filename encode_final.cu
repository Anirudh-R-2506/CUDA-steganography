#include <fstream>
#include <highgui.h>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
using namespace cv;

__global__ void LSB(unsigned char *input, char *message, int message_size) {
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  int offset = xIndex + yIndex * blockDim.x * gridDim.x;

  int charno = offset / 8;
  if (charno >= message_size) {
    return;
  }
  int bit_count = 7 - (offset % 8);
  char ch = message[charno] >> bit_count;
  if (ch & 1) {
    input[offset] |= 1;
  } else {
    input[offset] &= ~1;
  }
}

int main(int argc, char **argv) {

  if (argc != 4) {
    cout << "Number of Arguments Error"
         << "\n";
    exit(-1);
  }

  Mat image = imread(argv[1]);
  if (image.empty()) {
    cout << "Load Image Error\n";
    exit(-1);
  }

  ifstream file;
  file.open(argv[2]);
  if (!file.is_open()) {
    cout << "File Error\n";
    exit(-1);
  }

  stringstream strStream;
  strStream << file.rdbuf();
  string str = strStream.str();
  char arr[str.length() + 1];
  cout << "load text file size is " << str.size() << "\n";
  strcpy(arr, str.c_str());
  const int ImageSize = image.step * image.rows;
  int message_size = str.size() + 1;
  if ((message_size)*8 > ImageSize * 3) {
    printf("The input text file is too big, choose a larger image");
  }

  cv::Mat output(image.rows, image.cols, CV_8UC3);
  unsigned char *d_input;
  char *message;
  cudaMalloc<unsigned char>(&d_input, ImageSize);
  cudaMalloc((void **)&message, message_size * sizeof(char));

  cudaMemcpy(d_input, image.ptr(), ImageSize, cudaMemcpyHostToDevice);
  cudaMemcpy(message, arr, message_size * sizeof(char), cudaMemcpyHostToDevice);

  const dim3 block(16, 16);
  const dim3 grid((image.cols + block.x - 1) / block.x,
                  (image.rows + block.y - 1) / block.y);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  LSB<<<grid, block>>>(d_input, message, message_size);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Encode Kernel execution time is:  %3.10f sec\n", elapsedTime / 1000);

  cudaMemcpy(output.ptr(), d_input, ImageSize, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(message);

  imwrite(argv[3], output);
  return 0;
}
