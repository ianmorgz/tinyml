#pragma once
#include <stdint.h>

void model_init(void);
int forward(const int8_t* input, const int8_t* output);

