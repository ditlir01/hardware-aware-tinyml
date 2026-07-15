/* example_main.c — host smoke test for the float Neuro-C engine. */

#include <stdio.h>
#include "neuroc.h"
#include "model_data.h"
#include "test_sample.h"


/* All RAM for inference is statically allocated — no malloc on the MCU. */
static float scratch_a[MODEL_MAX_LAYER_WIDTH];
static float scratch_b[MODEL_MAX_LAYER_WIDTH];
static float output[MODEL_OUTPUT_SIZE];


int main(void) {
    printf("Neuro-C C port — single-sample inference demo\n");
    printf("  input_size      : %u\n", (unsigned)MODEL_INPUT_SIZE);
    printf("  output_size     : %u\n", (unsigned)MODEL_OUTPUT_SIZE);
    printf("  max_layer_width : %u\n", (unsigned)MODEL_MAX_LAYER_WIDTH);
    printf("  num_layers      : %u\n", (unsigned)MODEL.num_layers);

    neuroc_forward(&MODEL, TEST_INPUT, output, scratch_a, scratch_b);

    int predicted = neuroc_argmax(output, MODEL_OUTPUT_SIZE);
    printf("\nPredicted class : %d\n", predicted);
    printf("True class      : %d\n", TEST_LABEL);
    printf("Match           : %s\n", predicted == TEST_LABEL ? "yes" : "no");

    /* Dump the full logits/probabilities so we can cross-check with Keras. */
    printf("\nOutput vector:\n");
    for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
        printf("  [%2d] = %.6f\n", i, (double)output[i]);
    }

    return predicted == TEST_LABEL ? 0 : 1;
}
