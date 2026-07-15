# Host build for the C port. Targets: make (float Neuro-C), make int8, make mlp.

CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -Wpedantic
LDFLAGS ?= -lm

# Float Neuro-C target — uses model_data.h
TARGET       = neuroc_demo
SOURCES      = neuroc.c example_main.c
HEADERS      = neuroc.h model_data.h test_sample.h

# int8 Neuro-C target — uses model_data_int8.h (paper-faithful deployment)
TARGET_INT8  = neuroc_int8_demo
SOURCES_INT8 = neuroc_int8.c example_main_int8.c
HEADERS_INT8 = neuroc_int8.h model_data_int8.h test_sample.h

# MLP target — uses mlp_data.h (float baseline for comparison)
TARGET_MLP   = mlp_demo
SOURCES_MLP  = mlp.c example_main_mlp.c
HEADERS_MLP  = mlp.h mlp_data.h test_sample.h

.PHONY: all run int8 mlp clean

all: $(TARGET)

# --- Float Neuro-C ---
$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

# --- int8 Neuro-C (paper-faithful) ---
$(TARGET_INT8): $(SOURCES_INT8) $(HEADERS_INT8)
	$(CC) $(CFLAGS) $(SOURCES_INT8) -o $(TARGET_INT8) $(LDFLAGS)

int8: $(TARGET_INT8)
	./$(TARGET_INT8)

# --- Float MLP baseline ---
$(TARGET_MLP): $(SOURCES_MLP) $(HEADERS_MLP)
	$(CC) $(CFLAGS) $(SOURCES_MLP) -o $(TARGET_MLP) $(LDFLAGS)

mlp: $(TARGET_MLP)
	./$(TARGET_MLP)

clean:
	rm -f $(TARGET) $(TARGET_INT8) $(TARGET_MLP) *.o
