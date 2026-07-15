/* neuroc.c float Neuro-C forward pass. See neuroc.h. */

#include "neuroc.h"
#include <math.h>
#include <string.h>


static void apply_activation(float* vec, int n, neuroc_activation_t act) {
    switch (act) {
        case NEUROC_ACT_NONE:
            break;

        case NEUROC_ACT_RELU:
            for (int i = 0; i < n; i++) {
                if (vec[i] < 0.0f) vec[i] = 0.0f;
            }
            break;

        case NEUROC_ACT_SOFTMAX: {
            /* Numerically stable softmax: subtract max before exp. */
            float vmax = vec[0];
            for (int i = 1; i < n; i++) {
                if (vec[i] > vmax) vmax = vec[i];
            }
            float vsum = 0.0f;
            for (int i = 0; i < n; i++) {
                vec[i] = expf(vec[i] - vmax);
                vsum += vec[i];
            }
            if (vsum > 0.0f) {
                float inv = 1.0f / vsum;
                for (int i = 0; i < n; i++) vec[i] *= inv;
            }
            break;
        }
    }
}


/* Forward pass through one layer. */
static void neuroc_layer_forward(const neuroc_layer_t* layer,
                                 const float* in,
                                 float* out) {
    const int M  = layer->output_size;
    const int B  = layer->num_blocks;
    const int BS = layer->block_size;

    for (int j = 0; j < M; j++) {
        float acc = 0.0f;

        for (int b = 0; b < B; b++) {
            const int      base     = b * BS;
            const uint32_t pos_base = layer->pos_block_offset[b];
            const uint32_t neg_base = layer->neg_block_offset[b];
            const uint16_t* pos_cp  = layer->pos_col_ptr + b * (M + 1);
            const uint16_t* neg_cp  = layer->neg_col_ptr + b * (M + 1);

            /* +1 connections in this block */
            uint16_t ps = pos_cp[j];
            uint16_t pe = pos_cp[j + 1];
            for (uint16_t p = ps; p < pe; p++) {
                acc += in[base + layer->pos_indices[pos_base + p]];
            }

            /* -1 connections in this block */
            uint16_t ns = neg_cp[j];
            uint16_t ne = neg_cp[j + 1];
            for (uint16_t q = ns; q < ne; q++) {
                acc -= in[base + layer->neg_indices[neg_base + q]];
            }
        }

        /* One float multiply per output neuron - the per-neuron scale w_j. */
        out[j] = layer->scale[j] * acc + layer->bias[j];
    }

    apply_activation(out, M, layer->activation);
}


void neuroc_forward(const neuroc_model_t* model,
                    const float* input,
                    float* output,
                    float* scratch_a,
                    float* scratch_b) {
    const int L = model->num_layers;
    if (L == 0) return;

    if (L == 1) {
        neuroc_layer_forward(&model->layers[0], input, output);
        return;
    }

    const float* cur_in  = input;
    float*       cur_out = scratch_a;

    for (int l = 0; l < L; l++) {
        if (l == L - 1) cur_out = output;
        neuroc_layer_forward(&model->layers[l], cur_in, cur_out);
        cur_in = cur_out;
        if (l < L - 2) {
            cur_out = (cur_out == scratch_a) ? scratch_b : scratch_a;
        }
    }
}


int neuroc_argmax(const float* vec, int n) {
    if (n <= 0) return -1;
    int best_i = 0;
    float best_v = vec[0];
    for (int i = 1; i < n; i++) {
        if (vec[i] > best_v) {
            best_v = vec[i];
            best_i = i;
        }
    }
    return best_i;
}
