import cyllama as cy


def test_lowlevel_simple(model_path):
    assert cy.simple(
        model_path=model_path,
        prompt="When did the universe begin?",
        n_predict = 32,
    )

