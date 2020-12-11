#ifndef _BACKPROPAGATION_H_
#define _BACKPROPAGATION_H_
#include <fstream>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cmath>

class Matrix {
public:
	double** v;
	int n, m;

	Matrix();
	Matrix(int _n, int _m);
	~Matrix();
	double* operator[](int i);
	const double* operator[](int i) const;
	const Matrix& operator=(const Matrix& a);
};

class Network {
public:
	struct LayerT {
		std::vector<double> x; // вход слоя
		std::vector<double> z; // активированный выход слоя
		std::vector<double> df; // производная функции активации слоя
	};
	LayerT* L; // значения на каждом слое
	Matrix* weights; // матрицы весов слоя
	std::vector<double*> deltas; // дельты ошибки на каждом слое
	int layersN; // число слоёв

	Network(const std::vector<int>& _layers_size);
	void RecordWeights();
	void SetWeights();
	void Forward(double*& input, int size);
	void Backward(double*& output, int size, double& error);
	void UpdateWeights(double alpha);
	void Train(double**& X, double**& Y, int Xsize, int _image_size, double alpha, double eps, int epochs);
};

#endif  // _BACKPROPAGATION_H_