
N_CONTEXT=2048
N_PREDICT=512
REPEAT_PENALTY=1.15
REPEAT_LAST_N=128
N_GPU_LAYERS=99

MODEL=models/Llama-3.2-1B-Instruct-Q8_0.gguf

chat() {
    python3 -m src.cyllama.chat \
        -m ${MODEL} \
        -c ${N_CONTEXT} \
        --n-gpu-layers ${REPEAT_LAST_N}
}

chat1() {
    python3 -m src.cyllama.cli \
        -m ${MODEL} \
        -c ${N_CONTEXT} \
        -n ${N_PREDICT} \
        --color -cnv \
        --repeat-penalty ${REPEAT_PENALTY} \
        --repeat-last-n ${REPEAT_LAST_N}
}

ask() {
    python3 -m src.cyllama.cli \
        -m ${MODEL} \
        -c ${N_CONTEXT} \
        -n ${N_PREDICT} \
        --repeat-penalty ${REPEAT_PENALTY} \
        --repeat-last-n ${REPEAT_LAST_N} \
        -p $1
}
