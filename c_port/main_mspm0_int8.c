/* main_mspm0_int8.c — int8 Neuro-C inference timer for MSPM0G3507. */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <ti/devices/msp/msp.h>

#include "neuroc_int8.h"
#include "model_data_int8.h"
#include "test_sample.h"


static int8_t  input_int8[MODEL_INPUT_SIZE];
static int8_t  scratch_a[MODEL_MAX_LAYER_WIDTH];
static int8_t  scratch_b[MODEL_MAX_LAYER_WIDTH];
static int32_t output_logits[MODEL_OUTPUT_SIZE];


static void systick_cycle_counter_init(void) {
    SysTick->LOAD = 0x00FFFFFFul;
    SysTick->VAL  = 0;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk
                  | SysTick_CTRL_ENABLE_Msk;
}

static inline uint32_t systick_read_cycles(void) {
    return (SysTick->LOAD - SysTick->VAL) & 0x00FFFFFFul;
}


static inline int8_t quantize_to_int8(float f, float scale) {
    int v = (int)lroundf(f / scale);
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int8_t)v;
}


int main(void) {
    systick_cycle_counter_init();

    printf("=== Neuro-C int8 inference on MSPM0G3507 (Cortex-M0+) ===\n");
    printf("Model: input %u  output %u  layers %u  max_width %u\n",
           (unsigned)MODEL_INPUT_SIZE,
           (unsigned)MODEL_OUTPUT_SIZE,
           (unsigned)MODEL.num_layers,
           (unsigned)MODEL_MAX_LAYER_WIDTH);

    /* Pre-quantize the float input to int8 (one-time cost, not part of
     * the timed inference). On a real sensor pipeline the data would
     * arrive in int8 directly from the ADC. */
    for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
        input_int8[i] = quantize_to_int8(TEST_INPUT[i], MODEL_INPUT_SCALE);
    }

    /* Time one inference. */
    SysTick->VAL = 0;
    uint32_t t0 = systick_read_cycles();
    neuroc_int8_forward(&MODEL, input_int8, output_logits, scratch_a, scratch_b);
    uint32_t t1 = systick_read_cycles();
    uint32_t cycles = (t1 - t0) & 0x00FFFFFFul;

    int predicted = neuroc_int8_argmax(output_logits, MODEL_OUTPUT_SIZE);

    printf("\n--- Result ---\n");
    printf("Predicted : %d\n", predicted);
    printf("True      : %d\n", (int)TEST_LABEL);
    printf("Match     : %s\n", predicted == (int)TEST_LABEL ? "YES" : "NO");

    printf("\n--- Latency ---\n");
    printf("Cycles    : %lu\n", (unsigned long)cycles);
    printf("ms @ 32 MHz default : %.3f\n", cycles / 32000.0);
    printf("ms @ 80 MHz max     : %.3f\n", cycles / 80000.0);
    printf("ms @ 8 MHz (paper)  : %.3f\n", cycles / 8000.0);

    while (1) { __NOP(); }
    return 0;
}
