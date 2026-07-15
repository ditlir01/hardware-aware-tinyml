/* mlp.h — float32 MLP baseline (textbook dense layers). */

#ifndef MLP_H
#define MLP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    MLP_ACT_NONE    = 0,
    MLP_ACT_RELU    = 1,
    MLP_ACT_SOFTMAX = 2,
} mlp_activation_t;


/* One fully-connected float layer. */
typedef struct {
    uint16_t input_size;
    uint16_t output_size;

    /* Row-major weight matrix of shape (input_size, output_size).
     * kernel[i * output_size + j] is the weight from input i to output j.
     * Length: input_size * output_size. */
    const float* kernel;

    /* Per-neuron bias. Length: output_size. */
    const float* bias;

    mlp_activation_t activation;
} mlp_layer_t;


typedef struct {
    uint16_t num_layers;
    uint16_t max_layer_width;       /* max(output_size) over all layers */
    const mlp_layer_t* layers;
} mlp_model_t;


/* Forward pass through the entire model.
 *
 *   input        — input vector, length model->layers[0].input_size
 *   output       — output vector, length model->layers[num_layers-1].output_size
 *   scratch_a, b — two ping-pong buffers, length >= model->max_layer_width
 */
void mlp_forward(const mlp_model_t* model,
                 const float* input,
                 float* output,
                 float* scratch_a,
                 float* scratch_b);


/* Convenience: index of the largest entry in `vec`. */
int mlp_argmax(const float* vec, int n);


#ifdef __cplusplus
}
#endif

#endif /* MLP_H */
