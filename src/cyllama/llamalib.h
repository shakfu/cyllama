#ifndef LLAMALIB_H
#define LLAMALIB_H

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <cmath>
#include <cstdio>
#include <memory>


std::string simple_prompt(
    const std::string model_path,
    const std::string prompt,
    const int n_predict = 512,
    const int n_ctx = 2048,
    bool disable_log = true,
    int n_threads = 4)
{
    common_params params;

    params.prompt = prompt;
    params.model.path = model_path;
    params.n_predict = n_predict;
    params.n_ctx = n_ctx;
    params.verbosity = -1;
    params.cpuparams.n_threads = n_threads;

    if (disable_log) {
        common_log_set_verbosity_thold(params.verbosity);
    }

    // if (!gpt_params_parse(0, nullptr, params, LLAMA_EXAMPLE_COMMON, nullptr)) {
    //     LOG_ERR("%s: error: unable to parse gpt params\n" , __func__);
    //     return std::string();
    // }

    common_init();


    // init LLM
    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model
    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model_ptr = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (model_ptr == NULL) {
        LOG_ERR("%s: error: unable to load model\n" , __func__);
        return std::string();
    }

    // initialize the context
    llama_context_params ctx_params = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model_ptr, ctx_params);
    if (ctx == NULL) {
        LOG_ERR("%s: error: failed to create the llama_context\n" , __func__);
        return std::string();
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // tokenize the prompt
    std::vector<llama_token> tokens_list;
    tokens_list = ::common_tokenize(ctx, params.prompt, true);
    const int _n_ctx    = llama_n_ctx(ctx);
    const int n_kv_req = tokens_list.size() + (n_predict - tokens_list.size());

    LOG("\n");
    LOG_INF("\n%s: n_predict = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_predict, _n_ctx, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > _n_ctx) {
        LOG_ERR("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
        LOG_ERR("%s:        either reduce n_predict or increase n_ctx\n", __func__);
        return std::string();
    }

    // print the prompt token-by-token
    LOG("\n");
    for (auto id : tokens_list) {
        LOG("%s", common_token_to_piece(ctx, id).c_str());
    }

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding

    llama_batch batch = llama_batch_init(512, 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); i++) {
        common_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        return std::string();
    }

    const struct llama_vocab * vocab_ptr = llama_model_get_vocab(model_ptr);

    // main loop

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    // std::cout << "batch.n_tokens: " << batch.n_tokens << std::endl;

    std::string results;

    const auto t_main_start = ggml_time_us();

    while (n_cur <= n_predict) {
        // sample the next token
        {
            const llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab_ptr, new_token_id) || n_cur == n_predict) {
                LOG("\n");
                break;
            }

            results += common_token_to_piece(ctx, new_token_id);

            // prepare the next batch
            common_batch_clear(batch);

            // push this new token for next evaluation
            common_batch_add(batch, new_token_id, n_cur, { 0 }, true);

            n_decode += 1;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s : failed to eval, return code %d\n", __func__, 1);
            return std::string();
        }
    }

    LOG("\n");

    const auto t_main_end = ggml_time_us();

    LOG_INF("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    LOG("\n");

    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);

    LOG("\n");

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model_ptr);

    llama_backend_free();

    return results;
}


#endif // LLAMALIB_H

