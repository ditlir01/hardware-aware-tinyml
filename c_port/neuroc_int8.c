/* neuroc_int8.c — int8 Neuro-C forward pass. See neuroc_int8.h. */

#include "neuroc_int8.h"


/* Apply per-neuron activation + saturate to int8.
 * Returns the int8 result. */
static inline int8_t apply_act_and_saturate_int8(int32_t v,
                                                 neuroc_int8_activation_t act) {
    if (act == NEUROC_INT8_ACT_RELU && v < 0) v = 0;
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int8_t)v;
}


/* Compute one layer's forward pass into an int8 output buffer.
 * Used for all layers EXCEPT the final classification layer. */
static void layer_forward_int8(const neuroc_int8_layer_t* layer,
                               const int8_t* in,
                               int8_t* out) {
    const int M  = layer->output_size;
    const int B  = layer->num_blocks;
    const int BS = layer->block_size;
    const uint8_t shift = layer->shift;
    const int32_t round_term = (shift > 0) ? (int32_t)1 << (shift - 1) : 0;

    for (int j = 0; j < M; j++) {
        int32_t acc = 0;

        for (int b = 0; b < B; b++) {
            const int      base     = b * BS;
            const uint32_t pos_base = layer->pos_block_offset[b];
            const uint32_t neg_base = layer->neg_block_offset[b];
            const uint16_t* pos_cp  = layer->pos_col_ptr + b * (M + 1);
            const uint16_t* neg_cp  = layer->neg_col_ptr + b * (M + 1);

            uint16_t ps = pos_cp[j];
            uint16_t pe = pos_cp[j + 1];
            for (uint16_t p = ps; p < pe; p++) {
                acc += in[base + layer->pos_indices[pos_base + p]];
            }
            uint16_t ns = neg_cp[j];
            uint16_t ne = neg_cp[j + 1];
            for (uint16_t q = ns; q < ne; q++) {
                acc -= in[base + layer->neg_indices[neg_base + q]];
            }
        }

        /* int32 multiply (hardware on M0+) + int32 add */
        int32_t scaled = acc * (int32_t)layer->scale_q[j] + layer->bias_q[j];
        /* round and arithmetic shift right */
        scaled = (scaled + round_term) >> shift;

        out[j] = apply_act_and_saturate_int8(scaled, layer->activation);
    }
}


/* Compute the FINAL layer's forward pass into an int32 logits buffer.
 * No int8 saturation, no ReLU — we want raw values for the subsequent
 * argmax. */
static void layer_forward_int32_logits(const neuroc_int8_layer_t* layer,
                                       const int8_t* in,
                                       int32_t* out) {
    const int M  = layer->output_size;
    const int B  = layer->num_blocks;
    const int BS = layer->block_size;
    const uint8_t shift = layer->shift;
    const int32_t round_term = (shift > 0) ? (int32_t)1 << (shift - 1) : 0;

    for (int j = 0; j < M; j++) {
        int32_t acc = 0;

        for (int b = 0; b < B; b++) {
            const int      base     = b * BS;
            const uint32_t pos_base = layer->pos_block_offset[b];
            const uint32_t neg_base = layer->neg_block_offset[b];
            const uint16_t* pos_cp  = layer->pos_col_ptr + b * (M + 1);
            const uint16_t* neg_cp  = layer->neg_col_ptr + b * (M + 1);

            uint16_t ps = pos_cp[j];
            uint16_t pe = pos_cp[j + 1];
            for (uint16_t p = ps; p < pe; p++) {
                acc += in[base + layer->pos_indices[pos_base + p]];
            }
            uint16_t ns = neg_cp[j];
            uint16_t ne = neg_cp[j + 1];
            for (uint16_t q = ns; q < ne; q++) {
                acc -= in[base + layer->neg_indices[neg_base + q]];
            }
        }

        int32_t scaled = acc * (int32_t)layer->scale_q[j] + layer->bias_q[j];
        scaled = (scaled + round_term) >> shift;
        out[j] = scaled;
    }
}


void neuroc_int8_forward(const neuroc_int8_model_t* model,
                         const int8_t* input,
                         int32_t* output_logits,
                         int8_t* scratch_a,
                         int8_t* scratch_b) {
    const int L = model->num_layers;
    if (L == 0) return;

    if (L == 1) {
        /* Edge case: only one layer, treat it as the final logits layer. */
        layer_forward_int32_logits(&model->layers[0], input, output_logits);
        return;
    }

    /* Hidden layers ping-pong between scratch_a and scratch_b. */
    const int8_t* cur_in  = input;
    int8_t*       cur_out = scratch_a;

    for (int l = 0; l < L - 1; l++) {
        layer_forward_int8(&model->layers[l], cur_in, cur_out);
        cur_in = cur_out;
        cur_out = (cur_out == scratch_a) ? scratch_b : scratch_a;
    }

    /* Final layer: int32 logits into output_logits. */
    layer_forward_int32_logits(&model->layers[L - 1], cur_in, output_logits);
}


int neuroc_int8_argmax(const int32_t* vec, int n) {
    if (n <= 0) return -1;
    int best_i = 0;
    int32_t best_v = vec[0];
    for (int i = 1; i < n; i++) {
        if (vec[i] > best_v) {
            best_v = vec[i];
            best_i = i;
        }
    }
    return best_i;
}
