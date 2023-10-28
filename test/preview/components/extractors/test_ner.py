import pytest

from haystack.preview import DeserializationError, Document, Pipeline
from haystack.preview.components.extractors import (
    NamedEntityAnnotation,
    NamedEntityExtractor,
    NamedEntityExtractorBackend,
)


@pytest.fixture
def raw_texts():
    return [
        "My name is Clara and I live in Berkeley, California.",
        "I'm Merlin, the happy pig!",
        "New York State declared a state of emergency after the announcement of the end of the world.",
        "",  # Intentionally empty.
    ]


@pytest.fixture
def hf_annotations():
    return [
        [
            NamedEntityAnnotation(entity="PER", start=11, end=16),
            NamedEntityAnnotation(entity="LOC", start=31, end=39),
            NamedEntityAnnotation(entity="LOC", start=41, end=51),
        ],
        [NamedEntityAnnotation(entity="PER", start=4, end=10)],
        [NamedEntityAnnotation(entity="LOC", start=0, end=14)],
        [],
    ]


@pytest.fixture
def spacy_annotations():
    return [
        [
            NamedEntityAnnotation(entity="PERSON", start=11, end=16),
            NamedEntityAnnotation(entity="GPE", start=31, end=39),
            NamedEntityAnnotation(entity="GPE", start=41, end=51),
        ],
        [NamedEntityAnnotation(entity="PERSON", start=4, end=10)],
        [NamedEntityAnnotation(entity="GPE", start=0, end=14)],
        [],
    ]


@pytest.mark.integration
def test_ner_extractor_init():
    extractor = NamedEntityExtractor(
        backend=NamedEntityExtractorBackend.HUGGING_FACE, model_name_or_path="dslim/bert-base-NER", device_id=-1
    )

    with pytest.raises(ValueError, match=r"not initialized"):
        extractor.run(documents=[])

    assert not extractor.initialized
    extractor.warm_up()
    assert extractor.initialized


@pytest.mark.unit
def test_ner_extractor_serde():
    extractor = NamedEntityExtractor(
        backend=NamedEntityExtractorBackend.HUGGING_FACE, model_name_or_path="dslim/bert-base-NER", device_id=-1
    )

    serde_data = extractor.to_dict()
    new_extractor = NamedEntityExtractor.from_dict(serde_data)

    assert type(new_extractor._backend) == type(extractor._backend)
    assert new_extractor._backend.model_name == extractor._backend.model_name
    assert new_extractor._backend.device_id == extractor._backend.device_id

    with pytest.raises(DeserializationError, match=r"Couldn't deserialize"):
        serde_data["init_parameters"].pop("backend")
        _ = NamedEntityExtractor.from_dict(serde_data)


@pytest.mark.integration
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("device_id", [-1, 0])
def test_ner_extractor_hf_backend(raw_texts, hf_annotations, batch_size, device_id):
    extractor = NamedEntityExtractor(
        backend=NamedEntityExtractorBackend.HUGGING_FACE, model_name_or_path="dslim/bert-base-NER", device_id=device_id
    )
    extractor.warm_up()

    _extract_and_check_predictions(extractor, raw_texts, hf_annotations, batch_size)


@pytest.mark.integration
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("device_id", [-1, 0])
def test_ner_extractor_spacy_backend(raw_texts, spacy_annotations, batch_size, device_id):
    extractor = NamedEntityExtractor(
        backend=NamedEntityExtractorBackend.SPACY, model_name_or_path="en_core_web_trf", device_id=device_id
    )
    extractor.warm_up()

    _extract_and_check_predictions(extractor, raw_texts, spacy_annotations, batch_size)


@pytest.mark.integration
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("device_id", [-1, 0])
def test_ner_extractor_in_pipeline(raw_texts, hf_annotations, batch_size, device_id):
    pipeline = Pipeline()
    pipeline.add_component(
        name="ner_extractor",
        instance=NamedEntityExtractor(
            backend=NamedEntityExtractorBackend.HUGGING_FACE,
            model_name_or_path="dslim/bert-base-NER",
            device_id=device_id,
        ),
    )

    outputs = pipeline.run(
        {"ner_extractor": {"documents": [Document(text=text) for text in raw_texts], "batch_size": batch_size}}
    )["ner_extractor"]["documents"]
    predicted = [doc.metadata[NamedEntityExtractor.metadata_key] for doc in outputs]
    _check_predictions(predicted, hf_annotations)


def _extract_and_check_predictions(extractor, texts, expected, batch_size):
    docs = [Document(text=text) for text in texts]
    outputs = extractor.run(documents=docs, batch_size=batch_size)["documents"]
    assert all(id(a) == id(b) for a, b in zip(docs, outputs))
    predicted = [doc.metadata[NamedEntityExtractor.metadata_key] for doc in outputs]

    _check_predictions(predicted, expected)


def _check_predictions(predicted, expected):
    assert len(predicted) == len(expected)
    for pred, exp in zip(predicted, expected):
        assert len(pred) == len(exp)

        for a, b in zip(pred, exp):
            assert a.entity == b.entity
            assert a.start == b.start
            assert a.end == b.end
