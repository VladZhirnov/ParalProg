#include <mpi.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <iomanip>

using namespace std;

struct MatrixDim
{
    int rows = 0;
    int cols = 0;
};

vector<vector<float>> readMatrix(const string &filename, MatrixDim &dim, int rank) {
    ifstream file(filename);
    vector<vector<float>> matrix;
    string line;
    int line_num = 0;
    int expected_cols = -1;

    if (!file.is_open())
    {
        if (rank == 0)
            cerr << "Rank 0: Error - Cannot open file: " << filename << endl;
        dim = {0, 0};
        return matrix;
    }

    while (getline(file, line))
    {
        line_num++;
        stringstream ss(line);
        vector<float> row;
        float val;
        while (ss >> val)
        {
            row.push_back(val);
        }

        if (row.empty())
        {
            continue;
        }

        if (matrix.empty())
        {
            expected_cols = row.size();
        }
        else if ((int)row.size() != expected_cols)
        {
            if (rank == 0)
            {
                cerr << "Rank 0: Error - Inconsistent number of columns at line " << line_num
                     << " in file " << filename << ". Expected " << expected_cols
                     << ", found " << row.size() << "." << endl;
            }
            dim = {0, 0};
            matrix.clear();
            return matrix;
        }
        matrix.push_back(row);
    }

    if (matrix.empty())
    {
        if (rank == 0)
            cerr << "Rank 0: Warning - No valid data read from file " << filename << endl;
        dim = {0, 0};
    }
    else
    {
        dim.rows = matrix.size();
        dim.cols = matrix[0].size();
    }
    return matrix;
}


bool writeMatrix(const vector<vector<float>> &matrix, const string &filename, int rank) {
    ofstream file(filename);
    if (!file.is_open())
    {
        if (rank == 0)
            cerr << "Rank 0: Error - Cannot open output file: " << filename << endl;
        return false;
    }
    for (const auto &row : matrix)
    {
        bool first = true;
        for (float val : row)
        {
            if (!first)
            {
                file << " ";
            }
            file << val;
            first = false;
        }
        file << "\n";
    }
    if (rank == 0)
        cout << "Rank 0: Result matrix written to " << filename << endl;
    return true;
}


vector<float> flatten(const vector<vector<float>> &matrix)
{
    if (matrix.empty() || matrix[0].empty())
        return {};
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    vector<float> flat_matrix;
    flat_matrix.reserve(rows * cols);
    for (const auto &row : matrix)
    {
        flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }
    return flat_matrix;
}


bool multiplyMatrices(int rank, int size, const string &filenameA, const string &filenameB, const string &filenameC, vector<int> &times) {
    vector<vector<float>> A_full, B_full;
    vector<float> A_flat, B_flat;
    vector<float> C_flat;
    vector<float> local_A_flat;
    vector<float> local_C_flat;

    MatrixDim dimA = {0, 0}, dimB = {0, 0};
    int A_rows = 0, A_cols = 0, B_rows = 0, B_cols = 0;

    chrono::high_resolution_clock::time_point start_time, end_time;

    if (rank == 0)
    {
        start_time = chrono::high_resolution_clock::now();

        A_full = readMatrix(filenameA, dimA, rank);
        B_full = readMatrix(filenameB, dimB, rank);

        if (dimA.rows == 0 || dimA.cols == 0 || dimB.rows == 0 || dimB.cols == 0)
        {
            cerr << "Rank 0: Error reading matrices or matrices are empty for files: "
                 << filenameA << ", " << filenameB << ". Skipping this run." << endl;

            int dims[3] = {-1, -1, -1};
            MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
            return false;
        }
        if (dimA.cols != dimB.rows)
        {
            cerr << "Rank 0: Matrix dimensions incompatible for multiplication. "
                 << "A(" << dimA.rows << "x" << dimA.cols << "), "
                 << "B(" << dimB.rows << "x" << dimB.cols << ") for files "
                 << filenameA << ", " << filenameB << ". Skipping this run." << endl;
            int dims[3] = {-1, -1, -1};
            MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
            return false;
        }
        cout << "Rank 0: Processing A(" << dimA.rows << "x" << dimA.cols << "), "
             << "B(" << dimB.rows << "x" << dimB.cols << ")" << endl;

        A_rows = dimA.rows;
        A_cols = dimA.cols;
        B_rows = dimB.rows;
        B_cols = dimB.cols;
        A_flat = flatten(A_full);
        B_flat = flatten(B_full);
        if (A_flat.empty() || B_flat.empty())
        {
            cerr << "Rank 0: Failed to flatten matrices. Skipping this run." << endl;
            int dims[3] = {-1, -1, -1};
            MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
            return false;
        }
    }
    int dims_bcast[3] = {A_rows, A_cols, B_cols};
    MPI_Bcast(dims_bcast, 3, MPI_INT, 0, MPI_COMM_WORLD);

    if (dims_bcast[0] < 0)
    {
        return false;
    }

    if (rank != 0)
    {
        A_rows = dims_bcast[0];
        A_cols = dims_bcast[1];
        B_cols = dims_bcast[2];
        B_rows = A_cols;
    }

    if (A_rows <= 0 || A_cols <= 0 || B_cols <= 0)
    {
        if (rank != 0)
            cerr << "Rank " << rank << ": Received invalid dimensions. Skipping run." << endl;

        return false;
    }
    if (rank != 0)
    {
        B_flat.resize(B_rows * B_cols);
    }
    MPI_Bcast(B_flat.data(), B_rows * B_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    vector<int> sendcounts(size);
    vector<int> displs(size);
    int rows_per_proc = A_rows / size;
    int extra_rows = A_rows % size;
    int my_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);

    if (my_rows <= 0 && A_rows > 0)
    {
        local_A_flat.clear();
    }
    else if (A_rows == 0)
    {
        local_A_flat.clear();
        my_rows = 0;
    }
    else
    {
        local_A_flat.resize(my_rows * A_cols);
    }
    if (rank == 0)
    {
        int current_displ = 0;
        for (int i = 0; i < size; ++i)
        {
            int rows_for_rank_i = rows_per_proc + (i < extra_rows ? 1 : 0);
            sendcounts[i] = rows_for_rank_i * A_cols;
            displs[i] = current_displ;
            current_displ += sendcounts[i];
        }

        if (current_displ != A_rows * A_cols && A_rows > 0)
        {
            cerr << "Rank 0: FATAL - Displacement calculation error for Scatterv. Total displacement "
                 << current_displ << ", expected " << A_rows * A_cols << endl;
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }
    MPI_Scatterv(
        (A_rows > 0 ? A_flat.data() : nullptr),
        sendcounts.data(),
        displs.data(),
        MPI_FLOAT,
        (my_rows > 0 ? local_A_flat.data() : nullptr),
        my_rows * A_cols,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD);

    local_C_flat.resize(my_rows * B_cols, 0.0f);

    if (my_rows > 0)
    {
        for (int i = 0; i < my_rows; ++i)
        {
            for (int j = 0; j < B_cols; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < A_cols; ++k)
                {
                    sum += local_A_flat[i * A_cols + k] * B_flat[k * B_cols + j];
                }
                local_C_flat[i * B_cols + j] = sum;
            }
        }
    }

    vector<int> recvcounts;
    vector<int> gather_displs;
    if (rank == 0)
    {
        recvcounts.resize(size);
        gather_displs.resize(size);
        int current_displ = 0;
        for (int i = 0; i < size; ++i)
        {
            int rows_for_rank_i = rows_per_proc + (i < extra_rows ? 1 : 0);
            recvcounts[i] = rows_for_rank_i * B_cols;
            gather_displs[i] = current_displ;
            current_displ += recvcounts[i];
        }

        if (current_displ != A_rows * B_cols && A_rows > 0)
        {
            cerr << "Rank 0: FATAL - Displacement calculation error for Gatherv. Total displacement "
                 << current_displ << ", expected " << A_rows * B_cols << endl;
            MPI_Abort(MPI_COMM_WORLD, 5);
        }
        C_flat.resize(A_rows * B_cols);
    }

    MPI_Gatherv(
        local_C_flat.data(),
        my_rows * B_cols,
        MPI_FLOAT,
        (A_rows > 0 ? C_flat.data() : nullptr),
        recvcounts.data(),
        gather_displs.data(),
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD);

    if (rank == 0)
    {
        if (A_rows > 0)
        {
            vector<vector<float>> C_result(A_rows, vector<float>(B_cols));
            for (int i = 0; i < A_rows; ++i)
            {
                for (int j = 0; j < B_cols; ++j)
                {
                    if ((i * B_cols + j) >= C_flat.size())
                    {
                        cerr << "Rank 0: FATAL - Out of bounds access during C reconstruction!" << endl;
                        MPI_Abort(MPI_COMM_WORLD, 6);
                    }
                    C_result[i][j] = C_flat[i * B_cols + j];
                }
            }
            if (!writeMatrix(C_result, filenameC, rank))
            {
                return false;
            }
        }
        else
        {
            cout << "Rank 0: Skipping output write as matrix dimensions were zero." << endl;
        }

        end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        cout << "Rank 0: Time for size " << std::setw(4) << A_rows << "x" << A_cols
             << ": " << duration.count() << " ms" << endl;
        times.push_back(duration.count());
    }
    return true;
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> matrix_sizes = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000};
    vector<int> times;

    for (int current_size : matrix_sizes)
    {
        if (rank == 0)
        {
            cout << "\n-----------------------------------------" << endl;
            cout << "Starting processing for size: " << current_size << "x" << current_size << endl;
            cout << "-----------------------------------------" << endl;
        }

        string fileA = "Matrix_1/matrix1_" + to_string(current_size) + ".txt";
        string fileB = "Matrix_2/matrix2_" + to_string(current_size) + ".txt";
        string fileC = "Output/output_" + to_string(current_size) + ".txt";

        bool success = multiplyMatrices(rank, size, fileA, fileB, fileC, times);

        MPI_Barrier(MPI_COMM_WORLD);
    }
    string file_time = "time_mpi_" + to_string(size) + ".txt";
    ofstream file(file_time);
    for (auto x : times)
    {
        file << x << ", ";
    }

    if (rank == 0)
    {
        cout << "\nAll specified matrix sizes processed." << endl;
    }

    MPI_Finalize();
    return 0;
}