#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <sstream>

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


vector<vector<float>> multiplyMatrices(const vector<vector<float>> &a, const vector<vector<float>> &b) {
    size_t a_rows = a.size();
    size_t a_cols = a[0].size();
    size_t b_cols = b[0].size();
    
    vector<vector<float>> result(a_rows, vector<float>(b_cols, 0));
    
    for (size_t i = 0; i < a_rows; ++i) {
        for (size_t j = 0; j < b_cols; ++j) {
            for (size_t k = 0; k < a_cols; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    
    return result;
}


int get_time(const string &file1, const string &file2, const string &out_file) {
    auto matrix1 = readMatrix(file1);
    auto matrix2 = readMatrix(file2);

    if (matrix1[0].size() != matrix2.size()) {
        cerr << "Error: Matrix sizes are not compatible for multiplication!" << endl;
        return -1;
    }

    auto start = high_resolution_clock::now();
    auto result = multiplyMatrices(matrix1, matrix2);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    writeMatrix(result, out_file);
    return duration.count();
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
    
    stringstream times_stream;
    times_stream << time_10 << " " << time_20 << " " << time_30 << " "
                << time_40 << " " << time_50 << " " << time_60 << " "
                << time_70 << " " << time_80 << " " << time_90 << " "
                << time_100 << " " << time_1000;
    
    ofstream time_file("times.txt");
    if (time_file.is_open()) {
        time_file << times_stream.str();
        time_file.close();
    } else {
        cerr << "Error: Failed to create file times.txt" << endl;
    }

    cout << "All operations completed. Results written to times.txt" << endl;
    return 0;
}