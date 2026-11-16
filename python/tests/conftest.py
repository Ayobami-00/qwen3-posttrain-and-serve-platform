import pytest

from qps.specs import RunSpec
from qps.store import Store


@pytest.fixture
def character_tokenizer():
    """Predictable token boundaries without model downloads or ML imports."""

    class Tokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.template_options = kwargs
            return [10, 11]

        def encode(self, text, **kwargs):
            self.response = text
            self.encode_options = kwargs
            return [1000 + ord(character) for character in text]

    return Tokenizer()


@pytest.fixture
def local_run(tmp_path):
    spec = RunSpec.model_validate(
        {
            "metadata": {"name": "unit-run"},
            "spec": {
                "model": {"id": str(tmp_path / "parent"), "source": "local"},
                "dataset": {"id": str(tmp_path / "data.jsonl"), "source": "local"},
                "training": {"dtype": "fp32"},
                "execution": {"profile": "local-cpu-smoke", "gpuCount": 0},
                "artifacts": {"root": str(tmp_path / "artifacts")},
            },
        }
    )
    store = Store(tmp_path / "state")
    run = store.create_run(spec)
    return store, run, spec


@pytest.fixture(scope="session")
def successful(tmp_path_factory):
    from qps.smoke import smoke

    directory = tmp_path_factory.mktemp("real-cpu-workflow")
    report = smoke(directory, verbose=False)
    return Store(directory), report
