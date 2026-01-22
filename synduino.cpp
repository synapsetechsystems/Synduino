#include <stdint.h>
#include "synduino.h"
#include <Arduino.h>

void startup() {
  Serial.begin(9600);
  Serial.println("Serial communication started.");
}

int16_t** createWeights(int16_t nodes[], int16_t layers) {
  int16_t** weights = (int16_t**)malloc((layers - 1) * sizeof(int16_t*));
  if (weights == NULL) {
    Serial.println("Memory allocation for weights failed!");
    return NULL;
  }

  for (int16_t i = 0; i < layers - 1; i++) {
    weights[i] = (int16_t*)malloc(nodes[i] * nodes[i + 1] * sizeof(int16_t));
    if (weights[i] == NULL) {
      Serial.println("Memory allocation for weights row failed!");
      return NULL;
    }
  }

  // Initialize weights with random values
  for (int16_t i = 0; i < layers - 1; i++) {
    for (int16_t j = 0; j < nodes[i] * nodes[i + 1]; j++) {
      weights[i][j] = random(-10, 10);
    }
  }

  return weights;
}

int16_t** createBiases(int16_t nodes[], int16_t layers) {
  int16_t** biases = (int16_t**)malloc((layers - 1) * sizeof(int16_t*));

  for (int16_t i = 0; i < layers - 1; i++) {
    biases[i] = (int16_t*)malloc(nodes[i + 1] * sizeof(int16_t));
    for (int16_t j = 0; j < nodes[i + 1]; j++) {
      biases[i][j] = random(-10, 10);
    }
  }

  return biases;
}

int16_t* matrix_multiply(int16_t* input, int16_t* weights, int16_t input_size, int16_t layer_size) {
    int16_t* output = (int16_t*)malloc(layer_size * sizeof(int16_t));
    const int SCALE = 0; // tune this to control precision (divide by 256)

    for (int16_t j = 0; j < layer_size; j++) {
        int32_t sum = 0;
        for (int16_t i = 0; i < input_size; i++) {
            sum += (int32_t)input[i] * (int32_t)weights[j * input_size + i];
            if (sum > 32767) sum = 32767;
            if (sum < -32768) sum = -32768;
        }
        // scale down to prevent overflow and cast back to int16_t
        output[j] = (int16_t)(sum >> SCALE);
    }

    return output;
}

int16_t* add_bias(int16_t* output, int16_t* biases, int16_t layer_size) {
  for (int16_t i = 0; i < layer_size; i++) {
    output[i] += biases[i];
  }
  return output;
}

int16_t* relu(int16_t* output, int16_t layer_size) {
  for (int16_t i = 0; i < layer_size; i++) {
    if (output[i] < 0) {
      output[i] = 0;
    }
  }
  
  return output;
}

int16_t* sigmoid(int16_t* output, int16_t layer_size) {
  for (int16_t i = 0; i < layer_size; i++) {
    float s = 1.0 / (1.0 + exp(-output[i]));
    output[i] = (int16_t)(s * 32767); // range 0–127 roughly
  }
  return output;
}

int16_t* halve(int16_t* output, int16_t layer_size) {
  for (int16_t i = 0; i < layer_size; i++) {
    output[i] = output[i] / 2;
  }
  return output;
}

int16_t* negate(int16_t* output, int16_t layer_size) {
  for (int16_t i = 0; i < layer_size; i++) {
    output[i] = -output[i];
  }
  return output;
}

int16_t* apply_activation(int16_t* output, int16_t layer_size, char activation) {
  switch (activation) {
    case ('n'): {
      break;
    }
    
    case ('r'): {
      output = relu(output, layer_size);
      break;
    };

    case ('s'): {
      output = sigmoid(output, layer_size);
      break;
    };

    case ('h'): {
      output = halve(output, layer_size);
      break;
    };

    case ('g'): {
      output = negate(output, layer_size);
      break;
    };
  }

  return output;
}

int16_t* forward(int16_t* input, int16_t** weights, int16_t** biases, int16_t* nodes, int16_t layers, char activations[]) {
  int16_t* output = input;
  //Serial.println("Forward Started.");
  
  for (int16_t i = 0; i < layers - 1; i++) {
    //Serial.print("Layer ");
    //Serial.println(i + 1);

    output = matrix_multiply(output, weights[i], nodes[i], nodes[i + 1]);
    output = add_bias(output, biases[i], nodes[i + 1]);

    switch (activations[i]) {
      case ('r'): {
        output = relu(output, nodes[i + 1]);
        break;
      };

      case ('s'): {
        output = sigmoid(output, nodes[i + 1]);
        break;
      };
    }

    //Serial.println(activations[i]);
  }

  return output;
}

int32_t MSE(int16_t* pred, int16_t* target, int16_t size, int32_t* grad) {
    int32_t sum = 0;  // wider type for accumulation
    for(int16_t i = 0; i < size; i++) {
        int32_t diff = (int32_t)pred[i] - (int32_t)target[i];
        if (diff > 1000) diff = 1000;
        if (diff < -1000) diff = -1000;
        sum += diff * diff;
        if (grad) grad[i] = diff;
    }
    return sum / size;
}

void SGD(Linear &layer, int16_t* x, int32_t* dx, int16_t lr) {
    // Update weights: W -= lr * dx * x^T
    for(int16_t i = 0; i < layer.output_size; i++) {
        for(int16_t j = 0; j < layer.input_size; j++) {
            int16_t grad = dx[i] * x[j];        // compute gradient (16-bit to prevent overflow)
            int32_t delta = (int32_t)grad * lr / 127;
            layer.weights[i * layer.input_size + j] -= (int16_t)delta;
        }
    }

    // Update biases: b -= lr * dx
    for(int16_t i = 0; i < layer.output_size; i++) {
        layer.biases[i] -= (int16_t)((int32_t)dx[i] * lr);
    }
}

void setones(Linear &layer) {
    for(int16_t i = 0; i < layer.output_size; i++) {
        for(int16_t j = 0; j < layer.input_size; j++) {
          layer.weights[i * layer.input_size + j] = 1;
        }
    }

    // Update biases: b -= lr * dx
    for(int16_t i = 0; i < layer.output_size; i++) {
      layer.biases[i] = 1;
    }
}

void mutate(Linear &layer, int lr = 1) {
  if (random(1, 2) == 1) {
    for(int16_t i = 0; i < layer.output_size; i++) {
        for(int16_t j = 0; j < layer.input_size; j++) {
            layer.weights[i * layer.input_size + j] += random(-lr, lr);
        }
    }
  }

  else {
    for(int16_t i = 0; i < layer.output_size; i++) {
        layer.biases[i] -= random(-lr, lr);
    }
  }
}

void adjust(Linear layers[], int num_layers, int lr = 1) {
  int layer_idx = random(0, num_layers);
  Linear &layer = layers[layer_idx];

  bool adjust_weights = (random(0, 2) == 0);

  if (adjust_weights) {
    int r = random(0, layer.input_size);
    int c = random(0, layer.output_size);
    int change = (random(0, 2) == 0) ? lr : -lr;

    layer.weights[r * layer.output_size + c] += change;
  } else {
    int b = random(0, layer.output_size);
    int change = (random(0, 2) == 0) ? lr : -lr;

    layer.biases[b] += change;
  }
}

Linear::Linear(int16_t in_, int16_t out, char activation) {
  input_size = in_;
  output_size = out;
        
  int16_t nodes[2] = {in_, out};
  weights = createWeights(nodes, 2)[0];
  biases  = createBiases(nodes, 2)[0];
}

int16_t* Linear::forward(int16_t* x) {
  int16_t* output = matrix_multiply(x, weights, input_size, output_size);
  output = add_bias(output, biases, output_size);
  output = apply_activation(output, output_size, activation);
  return output;
}

int16_t* Linear::backward(int16_t* dx) {
    for(int16_t i = 0; i < output_size; i++) {
        switch(activation) {
          case 'r':  // ReLU
              dx[i] = (dx[i] > 0 ? dx[i] : 0);
              break;
  
          case 's':  // Sigmoid approx
              dx[i] = (dx[i] * (127 - dx[i])) / 127;
              break;
  
          case 'h':  // Halve
              dx[i] = dx[i] / 2;
              break;
  
          case 'g':  // Negate
              dx[i] = -dx[i];
              break;
  
          case 'n':  // No activation
              break;
      }
    }

    int16_t* dx_prev = (int16_t*)malloc(input_size * sizeof(int16_t));
    for(int16_t j = 0; j < input_size; j++) {
        int16_t grad = 0;
        for(int16_t i = 0; i < output_size; i++) {
            grad += dx[i] * weights[i * input_size + j];
        }
        dx_prev[j] = grad;
    }

    return dx_prev;
}

Sequential::Sequential(Linear* layers, int num_layers) {

}

int16_t* Sequential::forward(int16_t* x) {
  for (int i = 0; i < num_layers; i++) {
    x = lineup[i].forward(x);
  }
  return x;
}

int16_t* Sequential::backward(int16_t* dx) {
  for (int i = 0; i < num_layers; i++) {
    dx = lineup[i].backward(dx);
  }
  return dx;
}

void Sequential::setones() {
  for (int i = 0; i < num_layers; i++) {
    Linear layer = lineup[i];
    
    for(int16_t i = 0; i < layer.output_size; i++) {
        for(int16_t j = 0; j < layer.input_size; j++) {
          layer.weights[i * layer.input_size + j] = 1;
        }
    }

    for(int16_t i = 0; i < layer.output_size; i++) {
      layer.biases[i] = 1;
    }
  }
}
