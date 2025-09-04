import cyllama as cy



def test_sampler_instance():
    sparams = cy.LlamaSamplerChainParams()
    sparams.no_perf = False
    smplr = cy.LlamaSampler(sparams)
    smplr.add_temp(1.5)
    smplr.add_top_k(10)
    smplr.add_top_p(0.9, 10)
    smplr.add_typical(0.9, 10)
    smplr.add_greedy()
