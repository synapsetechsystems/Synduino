#ifndef SYNDUINO_H
#define SYNDUINO_H

void startup();

int16_t** createWeights(int16_t nodes[], int16_t layers);
int16_t** createBiases(int16_t nodes[], int16_t layers);

int16_t* matrix_multiply(int16_t* input, int16_t* weights, int16_t input_size, int16_t layer_size);
int16_t* add_bias(int16_t* output, int16_t* biases, int16_t layer_size);

int16_t* relu(int16_t* output, int16_t layer_size);
int16_t* sigmoid(int16_t* output, int16_t layer_size);
int16_t* halve(int16_t* output, int16_t layer_size);
int16_t* negate(int16_t* output, int16_t layer_size);
int16_t* apply_activation(int16_t* output, int16_t layer_size, char activation);

int16_t* forward(int16_t* input, int16_t** weights, int16_t** biases, int16_t* nodes, int16_t layers, char activations[]);

int32_t MSE(int16_t* pred, int16_t* target, int16_t size, int32_t* grad);

class Linear {
public:
  int16_t* weights;
  int16_t* biases;
  int16_t input_size;
  int16_t output_size;
  char activation;

  Linear(int16_t in_, int16_t out, char activation = 'n');

  int16_t* forward(int16_t* x);
  int16_t* backward(int16_t* dx);
};

void SGD(Linear &layer, int16_t* x, int32_t* dx, int16_t lr);
void setones(Linear &layer);
void mutate(Linear &layer, int lr = 1);
void adjust(Linear layers[], int num_layers, int lr = 1);

class Sequential {
public:
  Linear* lineup;
  int num_layers;

  Sequential(Linear* layers, int num_layers);
  
  int16_t* forward(int16_t* x);
  int16_t* backward(int16_t* dx);
  void setones();
};
#endif
