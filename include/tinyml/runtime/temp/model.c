#include "model.h"
#include "model_config.h"
#include "model_data.h"
#include "arena.h"
#include <stdio.h>

#define ARENA_BYTES 4096 // default

static uint8_t arena_buf[ARENA_BYTES];
static Arena arena;
static ModelCtx ctx;

void model_init(void) {
    arena_init(&arena, arena_buf, ARENA_BYTES);

    if(model_init_ctx(&model, &ctx, &arena) < 0){
        while(1){ printf("Error\n"); };
    }
}


void forward(const float* in, float* out) {
	if(model_forward(&model, &ctx, in, out) < 0){
		while(1) {
			printf("Error\n");
		}
	}
}

