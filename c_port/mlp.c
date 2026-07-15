/* mlp.c — float MLP forward pass. See mlp.h. */

#include "mlp.h"
#include <math.h>


static void apply_activation(float* vec, int n, mlp_activation_t act) {
    switch (act) {
        case MLP_ACT_NONE:
            break;

        case MLP_ACT_RELU:
            for (int i = 0; i < n; i++) {
                if (vec[i] < 0.0f) vec[i] = 0.0f;
            }
            break;

        case MLP_ACT_SOFTMAX: {
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


static void mlp_layer_forward(const mlp_layer_t* layer,
                              const float* in,
                              float* out) {
    const int N = layer->input_size;
    const int M = layer->output_size;
    const float* W = layer->kernel;

    /* Initialise outputs to biases. */
    for (int j = 0; j < M; j++) out[j] = layer->bias[j];

    /* Dense matmul: out += W^T * in (row-major W of shape [N, M]).
     * Outer loop over inputs is cache-friendly for sequential weight
     * fetch (each row of W is contiguous in flash). */
    for (int i = 0; i < N; i++) {
        const float xi = in[i];
        if (xi == 0.0f) continue;   /* skip work when input is exactly 0 */
        const float* row = W + i * M;
        for (int j = 0; j < M; j++) {
            out[j] += row[j] * xi;
        }
    }

    apply_activation(out, M, layer->activation);
}


void mlp_forward(const mlp_model_t* model,
                 const float* input,
                 float* output,
                 float* scratch_a,
                 float* scratch_b) {
    const int L = model->num_layers;
    if (L == 0) return;

    if (L == 1) {
        mlp_layer_forward(&model->layers[0], input, output);
        return;
    }

    const float* cur_in  = input;
    float*       cur_out = scratch_a;

    for (int l = 0; l < L; l++) {
        if (l == L - 1) cur_out = output;
        mlp_layer_forward(&model->layers[l], cur_in, cur_out);
        cur_in = cur_out;
        if (l < L - 2) {
            cur_out = (cur_out == scratch_a) ? scratch_b : scratch_a;
        }
    }
}


int mlp_argmax(const float* vec, int n) {
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
