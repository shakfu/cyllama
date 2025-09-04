import sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))

import cyllama as cy


def test_lowlevel_simple(model_path):
    assert cy.simple(
        model_path=model_path,
        prompt="When did the universe begin?",
        n_predict = 32,
    )







if __name__ == '__main__':
    test_lowlevel_simple('models/Llama-3.2-1B-Instruct-Q8_0.gguf')
