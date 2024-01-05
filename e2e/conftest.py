from pathlib import Path
from importlib.util import find_spec

import pytest
from haystack.testing.test_utils import set_all_seeds

from .pipelines.test_named_entity_extractor import SPACY_TEST_MODEL_NAME


set_all_seeds(0)


@pytest.fixture
def samples_path():
    return Path(__file__).parent / "samples"


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    # Set up additional dependencies

    # NamedEntityExtractor
    import spacy

    # The download function doesn't check it see if the model
    # is already installed before downloading the remote asset.
    # We can avoid this behaviour by checking its presence on our end.
    if find_spec(SPACY_TEST_MODEL_NAME) is None:
        spacy.cli.download(SPACY_TEST_MODEL_NAME)
