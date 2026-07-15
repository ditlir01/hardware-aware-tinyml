/* example_main_mlp.c — host smoke test for the float MLP engine. */

#include <stdio.h>
#include "mlp.h"
#include "mlp_data.h"
#include "test_sample.h"


static float scratch_a[MLP_MODEL_MAX_LAYER_WIDTH];
static float scratch_b[MLP_MODEL_MAX_LAYER_WIDTH];
static float output[MLP_MODEL_OUTPUT_SIZE];


int main(void) {
    printf("MLP host smoke test\n");
    printf("  input_size      : %u\n", (unsigned)MLP_MODEL_INPUT_SIZE);
    printf("  output_size     : %u\n", (unsigned)MLP_MODEL_OUTPUT_SIZE);
    printf("  max_layer_width : %u\n", (unsigned)MLP_MODEL_MAX_LAYER_WIDTH);
    printf("  num_layers      : %u\n", (unsigned)MLP_MODEL.num_layers);

    mlp_forward(&MLP_MODEL, TEST_INPUT, output, scratch_a, scratch_b);

    int predicted = mlp_argmax(output, MLP_MODEL_OUTPUT_SIZE);
    printf("\nPredicted class : %d\n", predicted);
    printf("True class      : %d\n", TEST_LABEL);
    printf("Match           : %s\n", predicted == TEST_LABEL ? "yes" : "no");

    printf("\nOutput vector:\n");
    for (int i = 0; i < MLP_MODEL_OUTPUT_SIZE; i++) {
        printf("  [%2d] = %.6f\n", i, (double)output[i]);
    }

    return predicted == TEST_LABEL ? 0 : 1;
}
