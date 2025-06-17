# import signal
# import sys
# from pathlib import Path

# from typing import Union, Optional

# Pathlike = Union[str, Path]

from . import cyllama as cy
# from . import log

def simple(model_path: str, prompt: str, ngl: int = 99, n_predict: int = 32):
    
    # load dynamic backends

    cy.ggml_backend_load_all()

    # initialize the model

    model_params = cy.LlamaModelParams()
    model_params.n_gpu_layers = ngl

    model = cy.LlamaModel(model_path, model_params)
    vocab = model.get_vocab()

    # tokenize the prompt

    # find the number of tokens in the prompt
    prompt_tokens = vocab.tokenize(prompt, add_special=True, parse_special=True)
    n_prompt = len(prompt_tokens)
    from IPython import embed; embed()


    # initialize the context

    ctx_params = cy.LlamaContextParams()
    # n_ctx is the context size
    ctx_params.n_ctx = n_prompt + n_predict - 1
    # n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt
    # enable performance counters
    ctx_params.no_perf = False

    ctx = cy.LlamaContext(model, ctx_params)

    # initialize the sampler

    sparams = cy.LlamaSamplerChainParams()
    sparams.no_perf = False

    smplr = cy.LlamaSampler(sparams)
    smplr.add_greedy()

    # llama_sampler_chain_add(smpl, llama_sampler_init_greedy())

    # print the prompt token-by-token

    # print the prompt token-by-token
    print()
    prompt=""
    for i in prompt_tokens:
        prompt += cy.common_token_to_piece(ctx, i)
    print(prompt)


    # prepare a batch for the prompt

    # make a static method of LlamaBatch
    batch: cy.LlamaBatch = cy.llama_batch_get_one(prompt_tokens)

    # main loop

    t_main_start: int = cy.ggml_time_us()
    n_decode = 0
    
    # llama_token new_token_id

    for n_pos in range(n_prompt + n_predict):
        ctx.decode(batch) # may raise ValueError

        n_pos += batch.n_tokens

        # sample the next token
        new_token_id = smplr.sample(ctx, -1)

        # is it an end of generation?
        if vocab.is_eog(new_token_id):
            break

        piece: str = vocab.token_to_piece(new_token_id, special=True)
        print(f"piece: %s", piece);

        # prepare the next batch with the sampled token
        batch = cy.llama_batch_get_one([new_token_id])

        n_decode += 1

    print()

    t_main_end: int = cy.ggml_time_us()

    print("decoded %d tokens in %.2f s, speed: %.2f t/s" %
            (n_decode, (t_main_end - t_main_start) / 1000000.0, n_decode / ((t_main_end - t_main_start) / 1000000.0)))
    print()

    smplr.print_perf_data()
    ctx.print_perf_data()

    assert True





# class Llama:
#     """top-level api class for llamalib"""

#     def __init__(
#         self,
#         model_path: Pathlike,
#         n_predict: int = 512,
#         n_ctx: int = 2048,
#         disable_log: bool = True,
#         n_threads: int = 4,
#     ):
#         self.model_path = Path(model_path)
#         self.disable_log = disable_log
#         self.log = log.config(self.__class__.__name__)
#         if not self.model_path.exists():
#             raise SystemExit(f"Provided model does not exist: {model_path}")

#         self.params = cy.CommonParams()
#         self.params.model = str(self.model_path)
#         self.params.n_predict = n_predict
#         self.params.n_ctx = n_ctx
#         self.params.verbosity = -1
#         self.params.cpuparams.n_threads = n_threads

#         self.model: Optional[cy.LlamaModel] = None
#         self.ctx: Optional[cy.LlamaContext] = None
#         self.smplr: Optional[cy.LlamaSampler] = None

#         self.chat_msgs: list[cy.CommonChatMsg] = []

#         self.path_session = Path(self.params.path_prompt_cache)
#         self.session_tokens = []
#         self.add_bos: bool = False

#         if self.disable_log:
#             cy.log_set_verbosity(self.params.verbosity)

#         cy.common_init()

#         # run configuration checks
#         self.check_params()

#         # init LLM
#         cy.llama_backend_init()
#         cy.llama_numa_init(self.params.numa)

#     def _termination_handler(self):
#         """handle termination by ctrl-c"""
#         print("TERMINATION...")
#         cy.llama_backend_free()
#         sys.exit(0)

#     def check_params(self):
#         if self.params.logits_all:
#             self.fail("please use the 'perplexity' tool for perplexity calculations")

#         if self.params.embedding:
#             self.fail("please use the 'embedding' tool for embedding calculations")

#         if self.params.n_ctx != 0 and self.params.n_ctx < 8:
#             self.log.warn("minimum context size is 8, using minimum size of 8.")
#             self.params.n_ctx = 8

#         if self.params.rope_freq_base != 0.0:
#             self.log.warn(
#                 "changing RoPE frequency base to %g", self.params.rope_freq_base
#             )

#         if self.params.rope_freq_scale != 0.0:
#             self.log.warn(
#                 "changing RoPE frequency base to %g", self.params.rope_freq_scale
#             )

#     def fail(self, msg, *args):
#         """exits the program with an error msg."""
#         self.log.critical(msg, *args)
#         sys.exit(1)

#     def attach_threadpool(self, ctx: cy.LlamaContext):
#         # MOVE this as context is defined later
#         tpp_batch = cy.GGMLThreadPoolParams.from_cpu_params(self.params.cpuparams_batch)
#         tpp = cy.GGMLThreadPoolParams.from_cpu_params(self.params.cpuparams)

#         cy.set_process_priority(self.params.cpuparams.priority)

#         if tpp.match(tpp_batch):
#             threadpool_batch = cy.GGMLThreadPool(tpp_batch)
#             tpp.paused = True

#             threadpool = cy.GGMLThreadPool(tpp)

#             ctx.attach_threadpool(threadpool, threadpool_batch)

#     def chat_add_and_format(
#         self, chat_msgs: list[cy.CommonChatMsg], role: str, content: str
#     ) -> str:
#         new_msg = cy.CommonChatMsg(role, content)
#         formatted = cy.common_chat_format_single(
#             self.model, self.chat_template, chat_msgs, new_msg, role == "user"
#         )
#         chat_msgs.append(CommonChatMsg(role, content))
#         self.log.debug("formatted: '%s'\n", formatted)
#         return formatted

#     def ask(
#         self, prompt: str, n_predict: Optional[int] = None, n_ctx: Optional[int] = None
#     ):
#         """prompt model"""

#         is_interacting = False
#         self.params.prompt = prompt

#         if n_predict:
#             self.params.n_predict = n_predict
#         if n_ctx:
#             self.params.n_ctx = n_ctx

#         # total length of the sequence including the prompt
#         n_predict: int = self.params.n_predict

#         # FIXME: PICK ONE, both ways work!
#         if 0:
#             # initialize the model
#             model_params = cy.common_model_params_to_llama(self.params)
#             self.model = cy.LlamaModel(
#                 path_model=self.params.model, params=model_params
#             )

#             # initialize the context
#             ctx_params = cy.common_context_params_to_llama(self.params)
#             self.ctx = cy.LlamaContext(model=self.model, params=ctx_params)
#         else:
#             # load the model and apply lora adapter, if any
#             llama_init = cy.CommonInitResult(self.params)
#             self.model = llama_init.model
#             self.ctx = llama_init.context

#         # attach threadpool
#         self.attach_threadpool(self.ctx)

#         # build sampler chain
#         sparams = cy.LlamaSamplerChainParams()
#         sparams.no_perf = False

#         self.smplr = cy.LlamaSampler(sparams)
#         self.smplr.add_greedy()

#         # tokenize the prompt
#         tokens_list: list[int] = cy.common_tokenize(self.ctx, self.params.prompt, True)
#         n_ctx_train: int = self.model.n_ctx_train
#         n_ctx: int = self.ctx.n_ctx

#         if n_ctx > n_ctx_train:
#             self.log.warn(
#                 "model was trained on only %d context tokens (%d specified)",
#                 n_ctx_train,
#                 n_ctx,
#             )

#         # print chat template example in conversation mode
#         if self.params.conversation:
#             if self.params.enable_chat_template:
#                 self.log.info(
#                     "chat template example:\n%s\n",
#                     cy.common_chat_format_example(
#                         self.model, self.params.chat_template
#                     ),
#                 )
#             else:
#                 self.log.info(
#                     "in-suffix/prefix is specified, chat template will be disabled."
#                 )

#         if not self.disable_log:
#             # print system information
#             self.log.info("\n%s\n", cy.common_params_get_system_info(self.params))

#         if self.path_session.name:  # is not empty
#             self.log.info(
#                 "attempting to load saved session from '%s'\n", self.path_session
#             )
#             if not self.path_session.exists():
#                 self.log.info("session file does not exist, will create.")
#             elif self.path_session.read_text() == "":
#                 self.log.info(
#                     "The session file is empty. A new session will be initialized."
#                 )
#             else:
#                 # The file exists and is not empty
#                 self.session_tokens = self.ctx.load_state_file(
#                     path_session=self.path_session, max_n_tokens=n_ctx
#                 )
#                 self.log.info(
#                     "loaded a session with prompt size of %d tokens",
#                     len(self.session_tokens),
#                 )

#         self.add_bos = self.model.add_bos_token()

#         if not self.model.has_encoder():
#             self.model.add_eos_token()

#         self.log.debug("n_ctx: %d, add_bos: %d", n_ctx, self.add_bos)

#         embd_inp = []

#         if (
#             self.params.conversation
#             and self.params.enable_chat_template
#             and not self.params.prompt
#         ):
#             # format the system prompt in conversation mode
#             prompt = self.chat_add_and_format(
#                 self.chat_msgs, "system", self.params.prompt
#             )
#         else:
#             prompt = self.params.prompt

#         if (
#             self.params.interactive_first
#             or not self.params.prompt
#             or not self.session_tokens
#         ):
#             self.log.debug("tokenize the prompt")
#             embd_inp = cy.common_tokenize(self.ctx, prompt, True, True)
#         else:
#             self.log.debug("use session tokens")
#             embd_inp = self.session_tokens

#         self.log.debug('prompt: "%s"', prompt)
#         self.log.debug("tokens: %s", cy.string_from_tokens(self.ctx, embd_inp))

#         # Should not run without any tokens
#         if not embd_inp:
#             if self.add_bos:
#                 embd_inp.append(self.model.token_bos())
#                 self.log.warn(
#                     "embd_inp was considered empty and bos was added: %s",
#                     cy.string_from_tokens(self.ctx, embd_inp),
#                 )
#             else:
#                 self.log.error("input is empty\n")
#                 raise SystemExit

#         # Tokenize negative prompt
#         if len(embd_inp) > n_ctx - 4:
#             self.log.error(
#                 "prompt is too long (%d tokens, max %d)", len(embd_inp), n_ctx - 4
#             )
#             raise SystemExit

#         # debug message about similarity of saved session, if applicable
#         n_matching_session_tokens = 0
#         if not self.session_tokens:
#             for token in self.session_tokens:
#                 if (
#                     n_matching_session_tokens >= len(embd_inp)
#                     or token != embd_inp[n_matching_session_tokens]
#                 ):
#                     break

#                 n_matching_session_tokens += 1

#             if not self.params.prompt and n_matching_session_tokens == len(embd_inp):
#                 self.log.info("using full prompt from session file")
#             elif n_matching_session_tokens >= len(embd_inp):
#                 self.log.info("session file has exact match for prompt!")
#             elif n_matching_session_tokens < (len(embd_inp) / 2):
#                 self.log.warning(
#                     "session file has low similarity to prompt (%d / %d tokens); will mostly be reevaluated",
#                     n_matching_session_tokens,
#                     len(embd_inp),
#                 )
#             else:
#                 self.log.info(
#                     "session file matches %d / %d tokens of prompt",
#                     n_matching_session_tokens,
#                     len(embd_inp),
#                 )

#             # remove any "future" tokens that we might have inherited from the previous session
#             self.ctx.kv_cache_seq_rm(-1, n_matching_session_tokens, -1)

#         self.log.debug(
#             "recalculate the cached logits (check): embd_inp.size() %d, n_matching_session_tokens %d, embd_inp.size() %d, session_tokens.size() %d",
#             len(embd_inp),
#             n_matching_session_tokens,
#             len(embd_inp),
#             len(self.session_tokens),
#         )

#         # if we will use the cache for the full prompt without reaching the end of the cache, force
#         # reevaluation of the last token to recalculate the cached logits
#         if (
#             not embd_inp
#             and n_matching_session_tokens == len(embd_inp)
#             and len(self.session_tokens) > len(embd_inp)
#         ):
#             self.log.debug(
#                 "recalculate the cached logits (do): session_tokens.resize( %d )",
#                 len(embd_inp) - 1,
#             )

#             self.session_tokens = self.session_tokens[: len(embd_inp) - 1]

#         # number of tokens to keep when resetting context
#         if self.params.n_keep < 0 or self.params.n_keep > len(embd_inp):
#             self.params.n_keep = len(embd_inp)
#         else:
#             self.params.n_keep += int(self.add_bos)  # always keep the BOS token

#         if self.params.conversation:
#             self.params.interactive_first = True

#         # enable interactive mode if interactive start is specified
#         if self.params.interactive_first:
#             self.params.interactive = True

#         if self.params.verbose_prompt:
#             self.log.info("prompt: '%s'", self.params.prompt)
#             self.log.info("number of tokens in prompt = %d", len(embd_inp))
#             for i in range(len(embd_inp)):
#                 self.log.info(
#                     "%d -> '%s'", embd_inp[i], self.ctx.token_to_piece(embd_inp[i])
#                 )

#             if self.params.n_keep > self.add_bos:
#                 self.log.info("static prompt based on n_keep: '")
#                 for i in range(self.params.n_keep):
#                     self.log.info("%s", self.ctx.token_to_piece(embd_inp[i]))
#                 self.log.info("")
#             self.log.info("")

#         # handle ctrl-c
#         signal.signal(signal.SIGINT, lambda signal, frame: self._termination_handler())

#         if self.params.interactive:
#             self.log.info("interactive mode on.\n")

#             if self.params.antiprompt:
#                 for antiprompt in self.params.antiprompt:
#                     self.log.info("Reverse prompt: '%s'", antiprompt)
#                     if self.params.verbose_prompt:
#                         tmp = self.ctx.tokenize(antiprompt, False, True)
#                         for i in range(len(tmp)):
#                             self.log.info(
#                                 "%6d -> '%s'\n", tmp[i], self.ctx.token_to_piece(tmp[i])
#                             )

#             if self.params.input_prefix_bos:
#                 self.log.info("Input prefix with BOS")

#             if self.params.input_prefix:
#                 self.log.info("Input prefix: '%s'", self.params.input_prefix)
#                 if self.params.verbose_prompt:
#                     tmp = self.ctx.tokenize(self.params.input_prefix, True, True)
#                     for i in range(len(tmp)):
#                         self.log.info(
#                             "%6d -> '%s'", tmp[i], self.ctx.token_to_piece(tmp[i])
#                         )

#             # this looks redundnat
#             if self.params.input_prefix:
#                 self.log.info("Input prefix: '%s'", self.params.input_prefix)
#                 if self.params.verbose_prompt:
#                     tmp = self.ctx.tokenize(self.params.input_prefix, False, True)
#                     for i in range(len(tmp)):
#                         self.log.info(
#                             "%6d -> '%s'", tmp[i], self.ctx.token_to_piece(tmp[i])
#                         )

#         sparams = self.params.sampling
#         smpl = self.model.sampler_init(sparams)
#         if not smpl:
#             self.fail("failed to initialize sampling subsystem\n")

#         self.log.info("sampler seed: %u", smpl.get_seed())
#         self.log.info("sampler params: %s", sparams.print())
#         self.log.info("sampler chain: %s", smpl.print())

#         self.log.info(
#             "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d",
#             n_ctx,
#             self.params.n_batch,
#             self.params.n_predict,
#             self.params.n_keep,
#         )

#         # group-attention state
#         # number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
#         ga_i: int = 0

#         ga_n: int = self.params.grp_attn_n
#         ga_w: int = self.params.grp_attn_w

#         if ga_n != 1:
#             assert ga_n > 0,"grp_attn_n must be positive"
#             assert ga_w % ga_n == 0, "grp_attn_w must be a multiple of grp_attn_n"
#             # assert n_ctx_train % ga_w == 0, "n_ctx_train must be a multiple of grp_attn_w"
#             # assert n_ctx >= n_ctx_train * ga_n, "n_ctx must be at least n_ctx_train * grp_attn_n"
#             self.log.info("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d", n_ctx_train, ga_n, ga_w)
#         self.log.info("")

#         if self.params.interactive:
#             if self.params.multiline_input:
#                 control_message = (
#                     " - To return control to the AI, end your input with '\\'.\n"
#                     " - To return control without starting a new line, end your input with '/'.\n"
#                 )
#             else:
#                 control_message = (
#                     " - Press Return to return control to the AI.\n"
#                     " - To return control without starting a new line, end your input with '/'.\n"
#                     " - If you want to submit another line, end your input with '\\'.\n"
#                 )
#             self.log.info("== Running in interactive mode. ==\n");
#             self.log.info(" - Press Ctrl+C to interject at any time.\n");
#             self.log.info("%s\n", control_message);

#             is_interacting = self.params.interactive_first;

#         is_antiprompt        : bool = False
#         input_echo           : bool = True
#         display              : bool = True
#         need_to_save_session : bool = self.path_session and n_matching_session_tokens < len(embd_inp)

#         n_past               : int = 0
#         n_remain             : int = self.params.n_predict
#         n_consumed           : int = 0
#         n_session_consumed   : int = 0

#         input_tokens: list[int] = []
#         g_input_tokens  = input_tokens

#         output_tokens: list[int] = []
#         g_output_tokens = output_tokens

#         # ostringstream output_ss
#         # g_output_ss     = &output_ss
#         # ostringstream assistant_ss # for storing current assistant message, used in conversation mode

#         # the first thing we will do is to output the prompt, so set color accordingly
#         # console::set_display(console::prompt)
#         display = self.params.display_prompt

#         embd: list[int] = [] # vector[llama_token]

#         # tokenized antiprompts
#         antiprompt_ids: list[list[int]] = [] # vector[vector[llama_token]]

#         for antiprompt in self.params.antiprompt:
#             antiprompt_ids.append(self.ctx.tokenize(antiprompt, False, True))

#         # if self.model.has_encoder():
#         #     enc_input_size: int = len(embd_inp)
#         #     llama_token * enc_input_buf = embd_inp.data()

#         #     if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size))):
#         #         self.fail("failed to eval\n")

#         #     decoder_start_token_id: cy.llama_token = self.model.decoder_start_token()
#         #     if (decoder_start_token_id == -1):
#         #         decoder_start_token_id = llama_token_bos(model)

#         #     embd_inp.clear()
#         #     embd_inp.push_back(decoder_start_token_id)



#         n_kv_req: int = len(tokens_list) + (n_predict - len(tokens_list))

#         if not self.disable_log:
#             self.log.info(
#                 "n_predict = %d, n_ctx = %d, n_kv_req = %d"
#                 % (n_predict, n_ctx, n_kv_req)
#             )

#         if n_kv_req > n_ctx:
#             self.fail(
#                 "error: n_kv_req > n_ctx, the required KV cache size is not big enough\n"
#                 "either reduce n_predict or increase n_ctx."
#             )

#         if not self.disable_log:
#             # print the prompt token-by-token
#             prompt = ""
#             for i in tokens_list:
#                 prompt += cy.common_token_to_piece(self.ctx, i)
#             self.log.info(prompt.strip())

#         # create a llama_batch with size 512
#         # we use this object to submit token data for decoding

#         # create batch
#         batch = cy.LlamaBatch(n_tokens=512, embd=0, n_seq_max=1)

#         # evaluate the initial prompt
#         for i, token in enumerate(tokens_list):
#             cy.common_batch_add(batch, token, i, [0], False)

#         # llama_decode will output logits only for the last token of the prompt
#         # batch.logits[batch.n_tokens - 1] = True
#         batch.set_last_logits_to_true()

#         # logits = batch.get_logits()

#         self.ctx.decode(batch)

#         # main loop

#         n_cur: int = batch.n_tokens
#         n_decode: int = 0
#         result: str = ""

#         if not self.disable_log:
#             t_main_start: int = cy.ggml_time_us()

#         while n_cur <= n_predict:
#             # sample the next token

#             if True:
#                 new_token_id = self.smplr.sample(self.ctx, batch.n_tokens - 1)

#                 self.smplr.accept(new_token_id)

#                 # is it an end of generation?
#                 if self.model.token_is_eog(new_token_id) or n_cur == n_predict:
#                     break

#                 result += cy.common_token_to_piece(self.ctx, new_token_id)

#                 # prepare the next batch
#                 cy.common_batch_clear(batch)
#                 # push this new token for next evaluation
#                 cy.common_batch_add(batch, new_token_id, n_cur, [0], True)

#                 n_decode += 1

#             n_cur += 1

#             # evaluate the current batch with the transformer model
#             self.ctx.decode(batch)

#         if not self.disable_log:
#             t_main_end: int = cy.ggml_time_us()
#             self.log.info(
#                 "decoded %d tokens in %.2f s, speed: %.2f t/s"
#                 % (
#                     n_decode,
#                     (t_main_end - t_main_start) / 1000000.0,
#                     n_decode / ((t_main_end - t_main_start) / 1000000.0),
#                 )
#             )

#         return result.strip()
