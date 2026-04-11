import cyllama.llama.llama_cpp as cy


def test_sampler_instance():
    sparams = cy.LlamaSamplerChainParams()
    sparams.no_perf = False
    smplr = cy.LlamaSampler(sparams)
    # Each add_* must return without raising and leave the sampler chain in
    # a state that still accepts further additions. A corrupted chain from
    # a broken add_* would raise on the next call.
    smplr.add_temp(1.5)
    smplr.add_top_k(10)
    smplr.add_top_p(0.9, 10)
    smplr.add_typical(0.9, 10)
    smplr.add_greedy()
    # The constructed sampler should still be the LlamaSampler type after
    # the chain additions.
    assert isinstance(smplr, cy.LlamaSampler)
