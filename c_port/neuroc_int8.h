/* neuroc_int8.h — int8 Neuro-C engine.
 * int8 activations + int32 accumulators; fixed-point per-neuron scale.
 * Inner loop: int adds/subs only — no float ops, no FPU needed.
 *   out_int8[j] = (int8_t)shifted;
 *
 * The final layer is left with activation=NONE; argmax is taken over
 * raw int32 logits via neuroc_int8_argmax(). Softmax is unnecessary —
 * argmax of softmax = argmax of logits.
 */

#ifndef NEUROC_INT8_H
#define NEUROC_INT8_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    NEUROC_INT8_ACT_NONE = 0,    /* leave raw int32 logits (for final layer) */
    NEUROC_INT8_ACT_RELU = 1,    /* clamp negative to 0 before saturating to int8 */
} neuroc_int8_activation_t;


/* One int8 Neuro-C fully-connected layer. */
typedef struct {
    uint16_t input_size;
    uint16_t output_size;
    uint16_t block_size;
    uint16_t num_blocks;

    /* Block-based ternary encoding, identical to neuroc.h. */
    const uint16_t* pos_col_ptr;
    const uint32_t* pos_block_offset;
    const uint8_t*  pos_indices;
    const uint16_t* neg_col_ptr;
    const uint32_t* neg_block_offset;
    const uint8_t*  neg_indices;

    /* Quantized per-neuron parameters. */
    const int16_t* scale_q;          /* length output_size; encoded with `shift` */
    const int32_t* bias_q;           /* length output_size; encoded with `shift` */
    uint8_t  shift;                  /* shared right-shift exponent for this layer */

    /* Activation applied AFTER the scale-and-shift step.
     * For hidden layers: NEUROC_INT8_ACT_RELU.
     * For the final classification layer: NEUROC_INT8_ACT_NONE (the
     * forward function writes int32 logits to the output buffer, not int8). */
    neuroc_int8_activation_t activation;
} neuroc_int8_layer_t;


typedef struct {
    uint16_t num_layers;
    uint16_t max_layer_width;
    const neuroc_int8_layer_t* layers;
} neuroc_int8_model_t;


/* Forward pass.
 *
 *   input          — int8 quantized input,  length layers[0].input_size
 *   output_logits  — int32 raw logits from the final layer, length
 *                    layers[last].output_size (used as argmax input)
 *   scratch_a,
 *   scratch_b      — two int8 ping-pong buffers, length >= max_layer_width
 *
 * Hidden layers write into the scratch buffers (int8). The final layer
 * (which has activation=NONE) writes int32 logits into output_logits.
 *
 * Either scratch buffer may alias the input or output if the caller wants
 * to skip a copy — same convention as the float version.
 */
void neuroc_int8_forward(const neuroc_int8_model_t* model,
                         const int8_t*  input,
                         int32_t*       output_logits,
                         int8_t*        scratch_a,
                         int8_t*        scratch_b);


/* Argmax over int32 logits — preserves softmax ordering without computing
 * the (expensive) exp() and division. */
int neuroc_int8_argmax(const int32_t* vec, int n);


#ifdef __cplusplus
}
#endif

#endif /* NEUROC_INT8_H */
