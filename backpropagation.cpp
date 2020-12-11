#include "backpropagation.h"

Matrix::Matrix() {
	n = 0;
	m = 0;
	v = nullptr;
};
Matrix::Matrix(int _n, int _m) { // n - кол-во нейронов выходного слоя, m - входного слоя
	n = _n;
	m = _m;
	srand((unsigned int)time(0));
	v = new double*[n];
	for (size_t i = 0; i < n; i++) {
		v[i] = new double[m];
		for (size_t j = 0; j < m; j++) {
			v[i][j] = (double)(rand()) / RAND_MAX - 0.5;
		}
	}
};
Matrix::~Matrix() {
	for (size_t i = 0; i < n; i++)
		delete[] v[i];
};
double* Matrix::operator[](int i) {
	return v[i];
};
const double* Matrix::operator[](int i) const {
	return v[i];
};
const Matrix& Matrix :: operator=(const Matrix& a) {
	if (this == &a)
		return *this;
	for (size_t i = 0; i < n; i++)
		delete[] v[i];
	n = a.n;
	m = a.m;
	v = new double* [n];
	for (size_t i = 0; i < n; i++) {
		v[i] = new double[m];
		for (size_t j = 0; j < m; j++)
			v[i][j] = a[i][j];
	}
	return *this;
};

Network::Network(const std::vector<int>& _layers_size) {
	srand((unsigned int)time(0));
	layersN = _layers_size.size() - 1;
	weights = new Matrix[layersN]; // создаём массив матриц весовых коэффициентов
	L = new LayerT[layersN]; // создаём массив значений на каждом слое
	deltas = std::vector<double*>(layersN); // создаём массив для дельт

	for (size_t k = 1; k < _layers_size.size(); k++) { // для каждого слоя создаём матрицы весовых коэффициентов
		weights[k - 1] = Matrix(_layers_size[k], _layers_size[k - 1]);
		L[k - 1].x = std::vector<double>(_layers_size[k - 1]); // создаём вектор для входа слоя
		L[k - 1].z = std::vector<double>(_layers_size[k]); // создаём вектор для выхода слоя
		L[k - 1].df = std::vector<double>(_layers_size[k]); // создаём вектор для производной слоя
		deltas[k - 1] = new double[_layers_size[k]]; // создаём вектор для дельт
	}
};
void Network::RecordWeights() {
	std::ofstream out("../Dataset/weights.txt"); 
	if (!out.is_open()) 
		throw "Can't open file";
	for (size_t k = 0; k < layersN; k++) {
		for (size_t i = 0; i < weights[k].n; i++) {
			for (size_t j = 0; j < weights[k].m; j++)
				out << weights[k][i][j] << " ";
		}
	}
	out.close();
};
void Network::SetWeights() {
	std::ifstream in("../Dataset/weights.txt");
	if (!in.is_open())
		throw "Can't open file";
	for (size_t k = 0; k < layersN; k++) {
		for (size_t i = 0; i < weights[k].n; i++) {
			for (size_t j = 0; j < weights[k].m; j++)
				in >> weights[k][i][j];
		}
	}
	in.close();
};
void Network::Forward(double*& input, int size) { // Прямое распространение	
	for (size_t k = 0; k < layersN; k++) { // проход по слоям
		// запись входного вектора
		if (k == 0) {
			for (size_t i = 0; i < size; i++)
				L[k].x[i] = input[i];
		} else {
			for (size_t i = 0; i < L[k - 1].z.size(); i++)
				L[k].x[i] = L[k - 1].z[i];
		}
		for (size_t i = 0; i < weights[k].n; i++) { // проход по нейронам выходного вектора
			double y = 0, expY = 0; // неактивированный выход нейрона
			for (size_t j = 0; j < weights[k].m; j++) { // проход по нейронам входного вектора
				y += weights[k][i][j] * L[k].x[j];
				expY += exp(y);
			}
            L[k].z[i] = 1 / (1 + exp(-y)); // активация с помощью сигмоидальной функции
            L[k].df[i] = L[k].z[i] * (1 - L[k].z[i]);
		}
	}
};
void Network::Backward(double*& output, int size, double& error) { // Обратное распространение
	int last = layersN - 1;
	error = 0;

	for (size_t i = 0; i < size; i++) { // вычисление дельты на последнем слое
		double e = L[last].z[i] - output[i]; // находим разность значений векторов
		deltas[last][i] = e * L[last].df[i]; // запоминаем дельту
		error += e * e / 2; // прибавляем к ошибке половину квадрата значения
	}

	// вычисляем каждую предудущю дельту на основе текущей с помощью умножения на транспонированную матрицу
	for (size_t k = last; k > 0; k--) { // проход по слоям
		for (size_t i = 0; i < weights[k].m; i++) { // проход по нейронам входного вектора
			deltas[k - 1][i] = 0;
			for (size_t j = 0; j < weights[k].n; j++) // проход по нейронам выходного вектора
				deltas[k - 1][i] += weights[k][j][i] * deltas[k][j];
			deltas[k - 1][i] *= L[k - 1].df[i]; // умножаем получаемое значение на производную предыдущего слоя
		}
	}
};
void Network::UpdateWeights(double alpha) { // обновление весовых коэффициентов, alpha - скорость обучения
	// Вычисляем градиент по весам и двигаемся в его отрицательную сторону
	for (size_t k = 0; k < layersN; k++) { // проход по слоям
		for (size_t i = 0; i < weights[k].n; i++) { // проход по нейронам выходного вектора
			for (size_t j = 0; j < weights[k].m; j++) // проход по нейронам входного вектора
				weights[k][i][j] -= alpha * deltas[k][i] * L[k].x[j]; // Градиент по весам равен перемножению входного вектора и вектора дельт
		}
	}
};
void Network::Train(double**& X, double**& Y, int Xsize, int _image_size, double alpha, double eps, int epochs) {
	int epoch = 1; // номер эпохи
	double error = 0; // ошибка эпохи
	std::cout << "Training..." << std::endl;
	do {
		for (size_t i = 0; i < Xsize; i++) { // проходимся по всем элементам обучающего множества
			Forward(X[i], _image_size); // прямое распространение сигнала
			Backward(Y[i], 10, error); // обратное распространение ошибки
			UpdateWeights(alpha); // обновление весовых коэффициентов
		}
		std::cout << "Epoch: " << epoch << ", Error: " << error << std::endl;
		epoch++; // увеличиваем номер эпохи

	} while (epoch <= epochs && error > eps);
}