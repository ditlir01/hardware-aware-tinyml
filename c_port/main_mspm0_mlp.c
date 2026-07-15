/* main_mspm0_mlp.c — float-MLP inference timer for MSPM0G3507. */

#include <stdio.h>
#include <stdint.h>
#include <ti/devices/msp/msp.h>

#include "mlp.h"
#include "mlp_data.h"
#include "test_sample.h"


/* Statically allocated buffers — all float. */
static float scratch_a[MLP_MODEL_MAX_LAYER_WIDTH];
static float scratch_b[MLP_MODEL_MAX_LAYER_WIDTH];
static float model_out[MLP_MODEL_OUTPUT_SIZE];


static void systick_cycle_counter_init(void) {
    SysTick->LOAD = 0x00FFFFFFul;
    SysTick->VAL  = 0;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk
                  | SysTick_CTRL_ENABLE_Msk;
}

static inline uint32_t systick_read_cycles(void) {
    return (SysTick->LOAD - SysTick->VAL) & 0x00FFFFFFul;
}


int main(void) {
    systick_cycle_counter_init();

    printf("=== MLP (float32) inference on MSPM0G3507 (Cortex-M0+) ===\n");
    printf("Model: input %u  output %u  layers %u  max_width %u\n",
           (unsigned)MLP_MODEL_INPUT_SIZE,
           (unsigned)MLP_MODEL_OUTPUT_SIZE,
           (unsigned)MLP_MODEL.num_layers,
           (unsigned)MLP_MODEL_MAX_LAYER_WIDTH);

    SysTick->VAL = 0;
    uint32_t t0 = systick_read_cycles();
    mlp_forward(&MLP_MODEL, TEST_INPUT, model_out, scratch_a, scratch_b);
    uint32_t t1 = systick_read_cycles();
    uint32_t cycles = (t1 - t0) & 0x00FFFFFFul;

    int predicted = mlp_argmax(model_out, MLP_MODEL_OUTPUT_SIZE);

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
