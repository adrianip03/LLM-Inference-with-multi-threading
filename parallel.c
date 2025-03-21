/*
* PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
* FILE NAME: parallel_3036064349.c
* NAME: Ip Hau Ching
* UID:  3036064349
* Development Platform: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
* Remark: 100%, I hope. (How much you implemented?)
* How to compile separately: (gcc -o parallel parallel_[UID].c -O2 -lm -lpthread)
*/
// grep "^mvm.*[0-9]" log | awk 'BEGIN{maxcol=0; maxrow=0}{count++; utime+=$2; if ($4>maxcol ) maxcol=$4; if ($5>maxrow) maxrow=$5} END {print count, utime, utime/count, maxcol, maxrow}'
// grep "^mha" log | awk '{count++; utime+=$2} END {print count, utime, utime/count}'
// grep "thr0" log | awk '{count++; utime+=$2} END {print count, utime, utime/count}'
// grep "thr0" log | awk '{print $1, $4}' | sort | uniq -c
// awk '{ if ($1 == "mvm:" && $2 == "invoked") invokeCount+=1; if ($1 == "mvm:" && $2 == "cond_wait_wakeup") wakeCount+=1} END {print "invokeCount: ", invokeCount, "wakeCount: ", wakeCount}' log
// grep "^thr.: mvm cond_wait_wakeup" log | sort | uniq -c
// 1536 49152

#include "common.h" // some common definitions

#include <unistd.h>       // for nearly everything :)
#include <stdio.h>        // for printf, sprintf, fgets
#include <stdlib.h>       // for malloc, calloc
#include <stdint.h>       // for uint8_t and uint64_t
#include <time.h>         // for time
#include <string.h>       // for memcpy and strcmp
#include <sys/resource.h> // for rusage collection

#include "model.h"// for Llama definitions -> no need to know

int pos = 0; // global position of generation
Transformer transformer; // transformer instance to be init
Tokenizer tokenizer;     // tokenizer instance to be init
Sampler sampler;         // sampler instance to be init

// YOUR CODE STARTS HERE
#include <pthread.h>
// #include <semaphore.h> // uncomment this line if you use semaphore
#include <stdbool.h>   // uncomment this line if you want true / false
#include <sys/syscall.h>

// you may define global variables here

typedef struct {
    pthread_t thread;
    bool working;
    int thr_id;
    struct rusage ru;
} ThreadInfo;

typedef struct {
    ThreadInfo *threads;
    int num_thr; 
    bool terminate;
    int num_done;
    pthread_cond_t finish;
    pthread_cond_t wake_up_do_shit;
    pthread_mutex_t lock;
    int task;   // task 1: mat_vec_mul      task 2: multi_head_attn
} ThreadPool;
ThreadPool *pool;

typedef struct {
    float* out;
    QuantizedTensor *vec;
    QuantizedTensor *mat;
    int col;
    int row;
} MatVecMulInfo;
MatVecMulInfo *mvminfo;

typedef struct {
    float* out;         // output tensor [head, head_size]
    float* q;           // query tensor  [head, head_size]
    float* key_cache;   // cache of history key tensor   [kv_head, seq_len, head_size]
    float* value_cache; // cache of history value tensor [kv_head, seq_len, head_size]
    float* att;         // buffer for attention score [head, seq_len]
    int seq_len;
    int n_heads;
    int head_size;
    int kv_dim;
    int kv_mul;
} MultiHeadAttnInfo;
MultiHeadAttnInfo *mhainfo;

// function executed by each thread to complete mat_vec_mul
// @note: please modify the signature to what you want
void mat_vec_mul_task_func(int thr_id) {
    

    int start_row;
    int end_row;
    if (mvminfo->row % pool->num_thr != 0){
        start_row = thr_id * mvminfo->row / (pool->num_thr -1);
        end_row = (thr_id + 1) * mvminfo->row / (pool->num_thr -1);
    } else {
        start_row = thr_id * mvminfo->row / pool->num_thr;
        end_row = (thr_id + 1) * mvminfo->row / pool->num_thr;
    }


    // Ensure the end_row does not exceed the total number of rows
    if (end_row > mvminfo->row) {
        end_row = mvminfo->row;
    }

    for (int i = start_row; i < end_row; i++) {

        float val = 0.0f; // final value
        int32_t ival = 0; // integer value to be dequantized
        int in = i * mvminfo->col;   // 

        // for each column
        // GS is group size of quantization, not included in assignment
        // @note please don't parallel this loop
        for (int j = 0; j <= mvminfo->col - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) mvminfo->vec->q[j + k]) * ((int32_t) mvminfo->mat->q[in + j + k]);
            }
            val += ((float) ival) * mvminfo->mat->s[(in + j) / GS] * mvminfo->vec->s[j / GS];
            ival = 0;
        }
        mvminfo->out[i] = val;
    }

}

// function executed by each thread to complete multi_head_attn
// @note: please modify the signature to what you want
void multi_head_attn_task_func(int thr_id) {
    
    int n = pool->num_thr;
    if (mvminfo->row % pool->num_thr != 0){
        n--;
    }

    int start_head = thr_id * mhainfo->n_heads / n;
    int end_head = (thr_id + 1) * mhainfo->n_heads / n;

    if (end_head > mhainfo->n_heads) {
        end_head = mhainfo->n_heads;
    }

    for (int h = start_head; h < end_head; h++) {
        // get the query vector for this head
        float* head_q = mhainfo->q + h * mhainfo->head_size;
        // attention scores for this head
        float* head_att = mhainfo->att + h * mhainfo->seq_len;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float* head_k = mhainfo->key_cache + t * mhainfo->kv_dim + (h / mhainfo->kv_mul) * mhainfo->head_size;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < mhainfo->head_size; i++) {
                score += head_q[i] * head_k[i];
            }
            score /= sqrtf(mhainfo->head_size);
            // save the score to the attention buffer
            head_att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(head_att, pos + 1);

        // weighted sum of the values, store back into xb
        float* head_out = mhainfo->out + h * mhainfo->head_size;
        memset(head_out, 0, mhainfo->head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float* head_v = mhainfo->value_cache + t * mhainfo->kv_dim + (h / mhainfo->kv_mul) * mhainfo->head_size;
            // get the attention weight for this timestep
            float a = head_att[t];
            // accumulate the weighted value into head out
            for (int i = 0; i < mhainfo->head_size; i++) {
                head_out[i] += a * head_v[i];
            }
        }
    }
}

// thread function used in pthread_create
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void *thr_func(void *arg) {
    ThreadInfo *info = (ThreadInfo *)arg;
    while (1) {

        pthread_mutex_lock(&pool->lock);
        while (!pool->terminate && !pool->threads[info->thr_id].working) { 
            pthread_cond_wait(&pool->wake_up_do_shit, &pool->lock);
        }
        if (pool->terminate) {
            pthread_mutex_unlock(&pool->lock);
            break;
        }
        pthread_mutex_unlock(&pool->lock);
        
        if (pool->task == 1) {
            mat_vec_mul_task_func(info->thr_id);    
        } else if (pool->task == 2) {
            multi_head_attn_task_func(info->thr_id);
        }
        // update num of threads that finished 
        pthread_mutex_lock(&pool->lock);
        pool->num_done++;
        pool->threads[info->thr_id].working = false;
        if (pool->num_done == pool->num_thr){
            pthread_cond_signal(&pool->finish);
        }
        pthread_mutex_unlock(&pool->lock);
        
    }

    getrusage(RUSAGE_THREAD, &info->ru);

    pthread_exit(NULL);
    
}

// function to initialize thread pool
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void init_thr_pool(int num_thr) {
    pool = (ThreadPool *)malloc(sizeof(ThreadPool));

    pool->threads = (ThreadInfo *)malloc(num_thr * sizeof(ThreadInfo));
    pool->num_thr = num_thr;
    pool->terminate = false;
    pool->num_done = 0;
    pool->task = 1;

    mvminfo = (MatVecMulInfo *)malloc(sizeof(MatVecMulInfo));

    mhainfo = (MultiHeadAttnInfo *)malloc(sizeof(MultiHeadAttnInfo));

    pthread_cond_init(&pool->finish, NULL);
    pthread_cond_init(&pool->wake_up_do_shit, NULL);
    pthread_mutex_init(&pool->lock, NULL);

    for (int i=0; i < num_thr; i++){
        pool->threads[i].thr_id = i;
        pool->threads[i].working = false;
        pthread_create(&pool->threads[i].thread, NULL, thr_func, &pool->threads[i]);

    }

}

// function to close thread pool
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void close_thr_pool() {
    
    pthread_mutex_lock(&pool->lock);
    pool->terminate = true;
    pthread_cond_broadcast(&pool->wake_up_do_shit);
    pthread_mutex_unlock(&pool->lock);

    for (int i=0; i < pool->num_thr; i++) {
        pthread_join(pool->threads[i].thread, NULL);
        fprintf(stdout, "\033[0;32mThread %d: user time = %.4f, system time = %.4f \033[0m\n", pool->threads[i].thr_id, (double)pool->threads[i].ru.ru_utime.tv_sec + (double)pool->threads[i].ru.ru_utime.tv_usec / 1000000.0, (double)pool->threads[i].ru.ru_stime.tv_sec + (double)pool->threads[i].ru.ru_stime.tv_usec / 1000000.0);
    }

    struct rusage ru;
    getrusage(RUSAGE_THREAD, &ru);
    fprintf(stdout, "\033[0;32mMain thread: user time = %.4f, system time = %.4f \033[0m\n", (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000.0, (double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec / 1000000.0);

    getrusage(RUSAGE_SELF, &ru);
    fprintf(stdout, "\033[0;32mProgram: user time = %.4f, system time = %.4f \033[0m\n", (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000.0, (double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec / 1000000.0);

    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->wake_up_do_shit);
    pthread_cond_destroy(&pool->finish);

    free(pool->threads);
    free(pool);
    free(mvminfo);
    free(mhainfo);

}

// ----------------------------------------------------------------------------
// entry function for multi-threading matrix multiplication
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void mat_vec_mul(float* out, QuantizedTensor *vec, QuantizedTensor *mat, int col, int row) {

    pthread_mutex_lock(&pool->lock);

    // load in mat_vec_mul info
    mvminfo->out = out;
    mvminfo->vec = vec;
    mvminfo->mat = mat;
    mvminfo->col = col;
    mvminfo->row = row;
    // wake up threads
    for (int i=0; i < pool->num_thr; i++){
        pool->threads[i].working = true;
    }
    pool->num_done = 0;
    pool->task = 1;
    pthread_cond_broadcast(&pool->wake_up_do_shit);

    // wait for all threads to finish
    if (pool->num_done != pool->num_thr) {
        pthread_cond_wait(&pool->finish, &pool->lock);
    }
    
    pthread_mutex_unlock(&pool->lock);
    
}

// ----------------------------------------------------------------------------
// entry function for multi-threading multi-head-attention
// @note: YOU CAN NOT MODIFY FUNCTION SIGNATURE!!!
void multi_head_attn(
    float* out,         // output tensor [head, head_size]
    float* q,           // query tensor  [head, head_size]
    float* key_cache,   // cache of history key tensor   [kv_head, seq_len, head_size]
    float* value_cache, // cache of history value tensor [kv_head, seq_len, head_size]
    float* att,         // buffer for attention score [head, seq_len]
    int seq_len,
    int n_heads,
    int head_size,
    int kv_dim,
    int kv_mul) {

    struct rusage ru;
    getrusage(RUSAGE_THREAD, &ru);
    double start_utime = (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000.0;
    double start_stime = (double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec / 1000000.0;

    pthread_mutex_lock(&pool->lock);

    // load in multi_head_attn info
    mhainfo->out = out;         
    mhainfo->q = q;           
    mhainfo->key_cache = key_cache;   
    mhainfo->value_cache = value_cache; 
    mhainfo->att = att;       
    mhainfo->seq_len = seq_len;
    mhainfo->n_heads = n_heads;
    mhainfo->head_size = head_size;
    mhainfo->kv_dim = kv_dim;
    mhainfo->kv_mul = kv_mul;

    // wake up threads
    for (int i=0; i < pool->num_thr; i++){
        pool->threads[i].working = true;
    }
    pool->num_done = 0;
    pool->task = 2;
    pthread_cond_broadcast(&pool->wake_up_do_shit);

    // wait for all threads to finish
    while (pool->num_done != pool->num_thr) {
        pthread_cond_wait(&pool->finish, &pool->lock);
    }
    pthread_mutex_unlock(&pool->lock);
    
    getrusage(RUSAGE_THREAD, &ru);
    double end_utime = (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000.0;
    double end_stime = (double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec / 1000000.0;

}
// YOUR CODE ENDS HERE

// ----------------------------------------------------------------------------
// forward Transformer, you're not allowed to modify this part
float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->q, &s->xq, w->wq + l, dim, dim);
        mat_vec_mul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        mat_vec_mul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        multi_head_attn(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, 
            p->seq_len, p->n_heads, head_size, kv_dim, kv_mul);

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        mat_vec_mul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        mat_vec_mul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    mat_vec_mul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// generation loop, you're not allowed to modify this part
void generate(char *prompt) {
    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+6) * sizeof(int)); // +6 reserved for prompt template
    encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;        // place holder for next token
    int token = prompt_tokens[0]; // place holder of prev token, kickoff as prompt_tokens[0]
    int end_pos = pos + MAX_NEW_TOKENS + num_prompt_tokens;
    int start_pos = pos;
    long start_time = 0; // to be lazy iniialzied
    while (pos < end_pos) {

        // forward the transformer to get logits for the next token
        float* logits = forward(&transformer, token, pos);

        if (pos < start_pos + num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos - start_pos + 1];
        } else if (pos == end_pos - 2) {
            // reaching the end, force it to close by <|im_end|>
            next = 2; // := <|im_end|>
        } else {
            // otherwise sample the next token from the logits
            next = sample(&sampler, logits);
        }

        pos++;

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(&tokenizer, token, next);
        if (pos >= num_prompt_tokens) {
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }

        token = next;

        // init the timer here because the first iteration can be slower
        if (start_time == 0) { start_time = time_in_ms(); }
    }
    printf("\n");

    long end_time = time_in_ms();
    // \033[0;32m set color to green and \033[0m reset to default, they won't be generate by LLM
    fprintf(stdout, "\033[0;32mlength: %d, speed (tok/s): %.4f \033[0m\n", 
        pos, (pos - start_pos) / (float) (end_time - start_time) * 1000);
    
    free(prompt_tokens);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *model_path     = "model.bin";  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature    = 0.6f;  // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp           = 0.9f;  // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    char *prompt         = NULL;  // prompt strings
    int num_prompt       = 0; // number of prompts
    uint64_t rng_seed    = 0; // seed rng with time by default
    int num_thr          = 0;

    if (argc == 4) {
        num_thr  = atoi(argv[1]);
        rng_seed = atoi(argv[2]);
        prompt   = argv[3];
    } else {
        fprintf(stderr, "Usage:   ./seq <num_thr> <seed> <prompt>\n");
        fprintf(stderr, "Example: ./seq 4 42 \"What is Fibonacci Number?\"\n");
        fprintf(stderr, "Note:    <prompt> must be quoted with \"\", only one prompt supported\n");
        exit(1);
    }

    // parameter validation/overrides
    if (num_thr <= 0 || num_thr > 16) { 
        fprintf(stderr, "num_thr must between 1 and 16 \n");
        exit(EXIT_FAILURE);
    }
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);

    // build the Transformer via the model .bin file
    build_transformer(&transformer, model_path);
    // build the Tokenizer via the tokenizer .bin file
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    // build the Sampler
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // initialize thread pool
    init_thr_pool(num_thr);

    printf("user: %s \n", prompt);
    // perform multi-threading generation
    generate(prompt);
    
    // close thread pool
    close_thr_pool();

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}