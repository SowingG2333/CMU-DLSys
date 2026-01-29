#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * 
     * 
     * 
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // 1. 计算迭代次数
    size_t num_iters = m / batch; 

    // 2. 提前申请空间
    float* Z = new float[batch * k];
    float* G = new float[n * k];

    // 3. batch迭代
    for (size_t i = 0; i < num_iters; i++) {
        // 梯度清零
        for (size_t j = 0; j < n * k; j++) {
            G[j] = 0;
        }
        // 计算当前 batch 在全量数据 X 中的起始行位置
        size_t batch_start_row = i * batch; 
        // 矩阵乘法计算：Z[batch, k] = X[batch, n] * theta[n, k]
        for (size_t r = 0; r < batch; r++) {        // 遍历当前 batch 的每一行
            for (size_t c = 0; c < k; c++) {        // 遍历 theta 的每一列
                float sum = 0.0;
                for (size_t l = 0; l < n; l++) {    // 累加维度 n
                    // X 的索引：(全局起始行 + 当前行) * 总列数 + 当前维度偏移
                    // theta 的索引：当前维度偏移 * 总列数 + 当前列
                    sum += X[(batch_start_row + r) * n + l] * theta[l * k + c];
                }
                // 存入 Z 的对应位置
                Z[r * k + c] = sum;
            }
        }
        // 计算Z矩阵的exp
        for (size_t j = 0; j < batch * k; j++) {
            Z[j] = std::exp(Z[j]);
        }
        // 进行归一化
        float row_sum = 0.0;
        for (size_t j = 0; j < batch; j++) {
            row_sum = 0.0;
            for (size_t l = 0; l < k; l++) {
                row_sum += Z[j * k + l];
            }
            for (size_t m = 0; m < k; m++) {
                Z[j * k + m] /= row_sum;
            }
        }
        // 直接在Z的基础上进行I_y的计算
        for (size_t j = 0; j < batch; j++) {
            Z[j * k + y[batch_start_row + j]] -= 1;
        }
        // 计算X.T * (Z - I_y)
        for (size_t f = 0; f < n; f++) {
            for (size_t c = 0; c < k; c++) {
                float sum = 0.0;
                for (size_t r = 0; r < batch; r++) {
                    // X 的行是 (batch_start_row + r)，列是 f
                    // Z 的行是 r，列是 c
                    sum += X[(batch_start_row + r) * n + f] * Z[r * k + c];
                }
                G[f * k + c] = sum;
            }
        }
        // 计算梯度
        for (size_t j = 0; j < n * k; j++) {
            G[j] /= batch;
            G[j] *= lr;
        }
        for (size_t j = 0; j < n * k; j++) {
            theta[j] -= G[j];
        }
    }
    // 3. 释放内存
    delete[] Z;
    delete[] G;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
