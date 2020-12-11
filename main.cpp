#include "backpropagation.h"

auto reverseInt = [](int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};

double** read_mnist_images(std::string path, int& _number_of_images, int& _image_size) {
    
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open())
        throw "Can't open file";

    int magic_number = 0, n_rows = 0, n_cols = 0;
    file.read((char*)&magic_number, sizeof(int)), magic_number = reverseInt(magic_number);
    if (magic_number != 2051) 
        throw "Invalid MNIST image file!";

    file.read((char*)&_number_of_images, sizeof(int)), _number_of_images = reverseInt(_number_of_images);
    file.read((char*)&n_rows, sizeof(int)), n_rows = reverseInt(n_rows);
    file.read((char*)&n_cols, sizeof(int)), n_cols = reverseInt(n_cols);

    _image_size = n_rows * n_cols;
    double** dataset = new double* [_number_of_images];
    for (size_t i = 0; i < _number_of_images; i++) {
        dataset[i] = new double[_image_size];
        for (size_t j = 0; j < _image_size; j++) {
            uint8_t pixel;
            file.read((char*)&pixel, 1);
            dataset[i][j] = double(pixel) / 255.;
        }
    }
    return dataset;
}

double** read_mnist_labels(std::string path, int& _number_of_labels) {

    std::ifstream file(path, std::ios::binary);

    if (!file.is_open())
        throw "Can't open file";

    int magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number)), magic_number = reverseInt(magic_number);
    if (magic_number != 2049) 
        throw std::runtime_error("Invalid MNIST label file!");
    
    file.read((char*)&_number_of_labels, sizeof(int)), _number_of_labels = reverseInt(_number_of_labels);

    double** dataset = new double* [_number_of_labels];
    for (size_t i = 0; i < _number_of_labels; i++) {
        uint8_t label;
        file.read((char*)&label, 1);
        double* direct_coding = new double[10]{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        direct_coding[int(label)] = 1;
        dataset[i] = direct_coding;
    }
    return dataset;
}


int main()
{
    int number_of_images, image_size;
    double** Xtrain, ** Ytrain, ** Xtest, ** Ytest;
    std::cout << "Reading data..." << std::endl;
    try {
        Xtrain = read_mnist_images("../Dataset/train-images.idx3-ubyte", number_of_images, image_size);
        Ytrain = read_mnist_labels("../Dataset/train-labels.idx1-ubyte", number_of_images);
        Xtest = read_mnist_images("../Dataset/t10k-images.idx3-ubyte", number_of_images, image_size);
        Ytest = read_mnist_labels("../Dataset/t10k-labels.idx1-ubyte", number_of_images);
    }
    catch (const char* s) {
        std::cout << s << std::endl;
        return -1;
    }
    std::vector<int> layers_size{ image_size, 512, 512, 10 };
    Network network(layers_size); // Создаём сеть с 784 входами, 512 нейронами в скрытых слоях и 10 выходами

    int mode = 0;
    do {
        std::cout << "Select a mode:" << std::endl;
        std::cout << "1 - reading the weights of a neural network trained on 1000 images from a file" << std::endl;
        std::cout << "2 - training a neural network" << std::endl;
        std::cout << "3 - writing the weights of the neural network to a file" << std::endl;
        std::cout << "4 - exit" << std::endl;
        std::cin >> mode;
        if (mode == 1) {
            network.SetWeights();
            for (int i = 0; i < 5; i++) {
                network.Forward(Xtest[i], image_size);
                for (int k = 0; k < 28; k++) {
                    for (int j = 0; j < 28; j++) {
                        if ((Xtest[i])[k * 28 + j] != 0)
                            std::cout << 1 << " ";
                        else std::cout << 0 << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << "Output: " << std::endl;
                for (int j = 0; j < 10; j++) {
                    double output = network.L[2].z[j];
                    std::cout << j << ": " << output << std::endl;
                }
            }
        }
        else if (mode == 2) {
            std::cout << "Enter the number of training pictures (0 < n <= 60 000)" << std::endl;
            int n = 0;
            while (n <= 0 || n > 60000) {
                std::cin >> n;
            }
            network.Train(Xtrain, Ytrain, n, image_size, 0.2, 0.01, 3);
            for (int i = 0; i < 5; i++) {
                network.Forward(Xtest[i], image_size);
                for (int k = 0; k < 28; k++) {
                    for (int j = 0; j < 28; j++) {
                        if ((Xtest[i])[k * 28 + j] != 0)
                            std::cout << 1 << " ";
                        else std::cout << 0 << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << "Output: " << std::endl;
                for (int j = 0; j < 10; j++) {
                    double output = network.L[2].z[j];
                    std::cout << j << ": " << output << std::endl;
                }
            }
        }
        else if (mode == 3) {
            std::cout << "Recording..." << std::endl;
            network.RecordWeights();
        }
        else if (mode == 4) return 0;
        else std::cout << "Invalid mode" << std::endl;
    } while (mode != 4);
}