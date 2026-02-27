#include "model.h"
#include "arena.h"
#include "model_config.h"
#include "model_data.h"

// generate this
#define ARENA_BYTES 4096 // default

static uint8_t arena_buf[ARENA_BYTES];
static Arena arena;
static ModelCtx ctx;

void model_init(void) {
    arena_init(&arena, arena_buf, ARENA_BYTES);

    if(!model_init_ctx(&model, &ctx, &arena)){
        while(1){ };
        // error in model validation
    }
}

int forward(const int8_t* input, const int8_t* output) {
    return model_forward(&model, &ctx, input, output);
}