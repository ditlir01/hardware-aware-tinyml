/* neuroc.h — float Neuro-C inference engine, block-based ternary encoding. */

#ifndef NEUROC_H
#define NEUROC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    NEUROC_ACT_NONE    = 0,   /* linear (use for raw logits on the final layer) */
    NEUROC_ACT_RELU    = 1,
    NEUROC_ACT_SOFTMAX = 2,   /* numerically stable softmax */
} neuroc_activation_t;


/* One Neuro-C fully-connected layer in block-based encoding. */
typedef struct {
    uint16_t input_size;
    uint16_t output_size;
    uint16_t block_size;        /* typically 256 — fits one byte of index range */
    uint16_t num_blocks;        /* = ceil(input_size / block_size) */

    /* +1 connections, grouped per block, then per output neuron. */
    const uint16_t* pos_col_ptr;        /* length num_blocks * (output_size + 1) */
    const uint32_t* pos_block_offset;   /* length num_blocks + 1 */
    const uint8_t*  pos_indices;        /* length pos_block_offset[num_blocks] */

    /* -1 connections, same layout. */
    const uint16_t* neg_col_ptr;
    const uint32_t* neg_block_offset;
    const uint8_t*  neg_indices;

    /* Per-neuron scaling factor w_j and bias b_j (Eq. 1). */
    const float* scale;                 /* length output_size */
    const float* bias;                  /* length output_size */

    neuroc_activation_t activation;
} neuroc_layer_t;


/* A whole stacked Neuro-C model. */
typedef struct {
    uint16_t num_layers;
    uint16_t max_layer_width;           /* max(output_size) over all layers — sizes scratch */
    const neuroc_layer_t* layers;       /* length num_layers */
} neuroc_model_t;


/* Forward pass.
 *
 *   input        — model input vector, length model->layers[0].input_size
 *   output       — model output, length model->layers[num_layers-1].output_size
 *   scratch_a,
 *   scratch_b    — two ping-pong buffers, length >= model->max_layer_width
 *
 * Either scratch buffer may alias `output`.
 */
void neuroc_forward(const neuroc_model_t* model,
                    const float* input,
                    float* output,
                    float* scratch_a,
                    float* scratch_b);


/* Convenience: return the index of the largest entry in `vec`.
 * Use as a stand-in for softmax on the final classification layer. */
int neuroc_argmax(const float* vec, int n);


#ifdef __cplusplus
}
#endif

#endif /* NEUROC_H */
