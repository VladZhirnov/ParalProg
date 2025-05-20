#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <time.h>

using namespace std;
using namespace std::chrono;


vector<vector<float>> readMatrix(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Failed to open file " << filename << endl;
        exit(1);
    }

    vector<vector<float>> matrix;
    string line;
    float num;

    while (getline(file, line)) {
        vector<float> row;
        istringstream iss(line);
        while (iss >> num) {
            row.push_back(num);
        }
        matrix.push_back(row);
    }

    file.close();
    return matrix;
}


void writeMatrix(const vector<vector<float>> &matrix, const string &filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Failed to open file " << filename << " for recording" << endl;
        exit(1);
    }

    for (const auto &row : matrix) {
        for (float val : row) {
            file << val << " ";
        }
        file << "\n";
    }
    
    file.close();
}


__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int A_rows, int A_cols, int B_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols)
    {
        float sum = 0.0f;
        for (int i = 0; i < A_cols; ++i)
        {
            sum += A[row * A_cols + i] * B[i * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}


vector<vector<float>> mulMatricesCUDA(const vector<vector<float>> &A, const vector<vector<float>> &B) {
    int A_rows = A.size(), A_cols = A[0].size(), B_cols = B[0].size();

    vector<float> h_A(A_rows * A_cols);
    vector<float> h_B(A_cols * B_cols);
    vector<float> h_C(A_rows * B_cols);

    for (int i = 0; i < A_rows; ++i)
        for (int j = 0; j < A_cols; ++j)
            h_A[i * A_cols + j] = A[i][j];

    for (int i = 0; i < A_cols; ++i)
        for (int j = 0; j < B_cols; ++j)
            h_B[i * B_cols + j] = B[i][j];

    float *d_A, *d_B, *d_C;
    size_t size_A = h_A.size() * sizeof(float);
    size_t size_B = h_B.size() * sizeof(float);
    size_t size_C = h_C.size() * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((B_cols + 15) / 16, (A_rows + 15) / 16);

    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, A_rows, A_cols, B_cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    vector<vector<float>> result(A_rows, vector<float>(B_cols));
    for (int i = 0; i < A_rows; ++i)
        for (int j = 0; j < B_cols; ++j)
            result[i][j] = h_C[i * B_cols + j];

    return result;
}


int get_time(const string &input1, const string &input2, const string &output) {
    auto start = chrono::high_resolution_clock::now();
    vector<vector<float>> vec1 = readMatrix(input1);
    vector<vector<float>> vec2 = readMatrix(input2);

    if (vec1.empty() || vec2.empty())
    {
        cerr << "Error: One or both input matrices are empty. Cannot multiply." << endl;
        return -1;
    }

    vector<vector<float>> result = mulMatricesCUDA(vec1, vec2);

    if (result.empty())
    {
        cerr << "Error: Error during multiplication" << endl;
        return -1;
    }
    writeMatrix(result, output);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Times: " << duration.count() << "\n";
    return duration.count();
}


void writeTime(const string &content, const string &name_file)
{
    ofstream file;
    file.open(name_file);
    file << content;
    file.close();
}


int main() {
    int time_10 = get_time("Matrix_1/matrix1_10.txt", "Matrix_2/matrix2_10.txt", "Output/output_10.txt");
    int time_20 = get_time("Matrix_1/matrix1_20.txt", "Matrix_2/matrix2_20.txt", "Output/output_20.txt");
    int time_30 = get_time("Matrix_1/matrix1_30.txt", "Matrix_2/matrix2_30.txt", "Output/output_30.txt");
    int time_40 = get_time("Matrix_1/matrix1_40.txt", "Matrix_2/matrix2_40.txt", "Output/output_40.txt");
    int time_50 = get_time("Matrix_1/matrix1_50.txt", "Matrix_2/matrix2_50.txt", "Output/output_50.txt");
    int time_60 = get_time("Matrix_1/matrix1_60.txt", "Matrix_2/matrix2_60.txt", "Output/output_60.txt");
    int time_70 = get_time("Matrix_1/matrix1_70.txt", "Matrix_2/matrix2_70.txt", "Output/output_70.txt");
    int time_80 = get_time("Matrix_1/matrix1_80.txt", "Matrix_2/matrix2_80.txt", "Output/output_80.txt");
    int time_90 = get_time("Matrix_1/matrix1_90.txt", "Matrix_2/matrix2_90.txt", "Output/output_90.txt");
    int time_100 = get_time("Matrix_1/matrix1_100.txt", "Matrix_2/matrix2_100.txt", "Output/output_100.txt");
    int time_1000 = get_time("Matrix_1/matrix1_1000.txt", "Matrix_2/matrix2_1000.txt", "Output/output_1000.txt");
    
    stringstream total_times;
    total_times << time_10 << ", " << time_20 << ", " << time_30 << ", " << time_40 << ", " << time_50 << ", " << time_60 << ", " << time_70 << ", " << time_80 << ", " << time_90 << ", " << time_100 << ", " << time_1000;
    string times = total_times.str();
    writeTime(times, "times_CUDA.txt");
    return 0;
}