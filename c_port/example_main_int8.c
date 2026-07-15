/* example_main_int8.c — host smoke test for the int8 Neuro-C engine. */

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "neuroc_int8.h"
#include "model_data_int8.h"
#include "test_sample.h"


/* Statically allocated buffers — no malloc. */
static int8_t  input_int8[MODEL_INPUT_SIZE];
static int8_t  scratch_a[MODEL_MAX_LAYER_WIDTH];
static int8_t  scratch_b[MODEL_MAX_LAYER_WIDTH];
static int32_t output_logits[MODEL_OUTPUT_SIZE];


/* Quantize a single float to int8 using the model's calibrated input scale. */
static inline int8_t quantize_to_int8(float f, float scale) {
    int v = (int)lroundf(f / scale);
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int8_t)v;
}


int main(void) {
    printf("Neuro-C int8 host smoke test\n");
    printf("  input_size      : %u\n", (unsigned)MODEL_INPUT_SIZE);
    printf("  output_size     : %u\n", (unsigned)MODEL_OUTPUT_SIZE);
    printf("  max_layer_width : %u\n", (unsigned)MODEL_MAX_LAYER_WIDTH);
    printf("  num_layers      : %u\n", (unsigned)MODEL.num_layers);
    printf("  input_scale     : %g\n", (double)MODEL_INPUT_SCALE);

    /* Quantize the float TEST_INPUT to int8 once. */
    for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
        input_int8[i] = quantize_to_int8(TEST_INPUT[i], MODEL_INPUT_SCALE);
    }

    neuroc_int8_forward(&MODEL, input_int8, output_logits, scratch_a, scratch_b);

    int predicted = neuroc_int8_argmax(output_logits, MODEL_OUTPUT_SIZE);

    printf("\nPredicted class : %d\n", predicted);
    printf("True class      : %d\n", TEST_LABEL);
    printf("Match           : %s\n", predicted == TEST_LABEL ? "yes" : "no");

    printf("\nOutput logits (int32, post-shift):\n");
    for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
        printf("  [%2d] = %ld\n", i, (long)output_logits[i]);
    }

    return predicted == TEST_LABEL ? 0 : 1;
}
