from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional, Set

from haystack import Document, Pipeline
from haystack.components.evaluators import (
    DocumentMAPEvaluator,
    DocumentMRREvaluator,
    DocumentRecallEvaluator,
    FaithfulnessEvaluator,
    SASEvaluator,
)
from haystack.components.evaluators.document_recall import RecallMode

from .eval_harness import EvaluationHarness
from .eval_run_result import EvaluationRunResult
from .util import PipelinePair, aggregate_batched_pipeline_outputs, deaggregate_batched_pipeline_inputs


class RAGComponent(Enum):
    """
    Represents the basic components in a RAG pipeline that need to be present for evaluation.

    Each of these can be separate components in the pipeline or a single component that performs
    multiple tasks.
    """

    #: The component in a RAG pipeline that accepts the user query.
    #: Expected inputs: `query` - Name of input that contains the query string.
    QUERY_PROCESSOR = "query_processor"

    #: The component in a RAG pipeline that retrieves documents based on the query.
    #: Expected outputs: `retrieved_documents` - Name of output containing retrieved documents.
    DOCUMENT_RETRIEVER = "document_retriever"

    #: The component in a RAG pipeline that generates responses based on the query and the retrieved documents.
    #: Expected outputs: `replies` - Name of out containing the LLM responses. Only the first response is used.
    RESPONSE_GENERATOR = "response_generator"


@dataclass(frozen=True)
class RAGComponentMetadata:
    """
    Metadata for a `RAGComponent`.

    :param name:
        Name of the component in the pipeline.
    :param input_mapping:
        Mapping of the expected inputs to
        corresponding component input names.
    :param output_mapping:
        Mapping of the expected outputs to
        corresponding component output names.
    """

    name: str
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)


class RAGMetric(Enum):
    """
    Represents the metrics that can be used to evaluate a RAG pipeline.
    """

    #: Document Mean Average Precision.
    DOCUMENT_MAP = "metric_doc_map"

    #: Document Mean Reciprocal Rank.
    DOCUMENT_MRR = "metric_doc_mrr"

    #: Document Recall with a single hit.
    DOCUMENT_RECALL_SINGLE_HIT = "metric_doc_recall_single"

    #: Document Recall with multiple hits.
    DOCUMENT_RECALL_MULTI_HIT = "metric_doc_recall_multi"

    #: Semantic Answer Similarity.
    SEMANTIC_ANSWER_SIMILARITY = "metric_sas"

    #: Answer Faithfulness.
    ANSWER_FAITHFULNESS = "metric_answer_faithfulness"


@dataclass(frozen=True)
class RAGEvalInput:
    """
    Input passed to the RAG evaluation harness.

    :param queries:
        The queries passed to the RAG pipeline.
    :param ground_truth_documents:
        The ground truth documents passed to the
        evaluation pipeline. Only required for metrics
        that require them.

        Corresponds to the queries.
    :param ground_truth_answers:
        The ground truth answers passed to the
        evaluation pipeline. Only required for metrics
        that require them.

        Corresponds to the queries.
    :param additional_rag_inputs:
        Additional inputs to pass to the RAG pipeline. Each
        key is the name of the component and its value a dictionary
        with the input name and a list of values, each corresponding
        to a query.
    """

    queries: List[str]
    ground_truth_documents: Optional[List[List[Document]]] = None
    ground_truth_answers: Optional[List[str]] = None
    additional_rag_inputs: Optional[Dict[str, Dict[str, List[Any]]]] = None


@dataclass(frozen=True)
class RAGEvalOverrides:
    """
    Overrides for a RAG evaluation run.

    Used to override the init parameters of components in
    either (or both) the evaluated and evaluation pipelines.

    :param rag:
        Overrides for the RAG pipeline. Each
        key is a component name and its value a dictionary
        with init parameters to override.
    :param eval:
        Overrides for the evaluation pipeline. Each
        key is a RAG metric and its value a dictionary
        with init parameters to override.
    """

    rag: Optional[Dict[str, Dict[str, Any]]] = None
    eval: Optional[Dict[RAGMetric, Dict[str, Any]]] = None


class RAGEvalHarness(EvaluationHarness[RAGEvalInput, RAGEvalOverrides, EvaluationRunResult]):
    def __init__(
        self, rag_pipeline: Pipeline, rag_components: Dict[RAGComponent, RAGComponentMetadata], metrics: Set[RAGMetric]
    ):
        """
        Create a evaluation harness for evaluating basic RAG pipelines.

        :param rag_pipeline:
            The RAG pipeline to evaluate.
        :param rag_components:
            A mapping of RAG components to their metadata.
        :param metrics:
            The metrics to use during evaluation.
        """
        super().__init__()

        self._validate_rag_components(rag_pipeline, rag_components)

        self.rag_pipeline = rag_pipeline
        self.rag_components = rag_components
        self.metrics = metrics
        self.evaluation_pipeline = self._build_evaluation_pipeline(metrics)

    @classmethod
    def default_with_embedding_retriever(cls, rag_pipeline: Pipeline, metrics: Set[RAGMetric]) -> "RAGEvalHarness":
        """
        Create a default evaluation harness for evaluating RAG pipelines with a query embedder.

        :param rag_pipeline:
            The RAG pipeline to evaluate. The following assumptions are made:
            - The query embedder component is named 'query_embedder' and has a 'text' input.
            - The document retriever component is named 'retriever' and has a 'documents' output.
            - The response generator component is named 'generator' and has a 'replies' output.
        :param metrics:
            The metrics to use during evaluation.
        """
        rag_components = {
            RAGComponent.QUERY_PROCESSOR: RAGComponentMetadata(name="query_embedder", input_mapping={"query": "text"}),
            RAGComponent.DOCUMENT_RETRIEVER: RAGComponentMetadata(
                name="retriever", output_mapping={"retrieved_documents": "documents"}
            ),
            RAGComponent.RESPONSE_GENERATOR: RAGComponentMetadata(
                name="generator", output_mapping={"replies": "replies"}
            ),
        }

        return cls(rag_pipeline, rag_components, deepcopy(metrics))

    @classmethod
    def default_with_keyword_retriever(cls, rag_pipeline: Pipeline, metrics: Set[RAGMetric]) -> "RAGEvalHarness":
        """
        Create a default evaluation harness for evaluating RAG pipelines with a keyword retriever.

        :param rag_pipeline:
            The RAG pipeline to evaluate. The following assumptions are made:
            - The document retriever component is named 'retriever' and has a 'query' input and a 'documents' output.
            - The response generator component is named 'generator' and has a 'replies' output.
        :param metrics:
            The metrics to use during evaluation.
        """
        rag_components = {
            RAGComponent.QUERY_PROCESSOR: RAGComponentMetadata(name="retriever", input_mapping={"query": "query"}),
            RAGComponent.DOCUMENT_RETRIEVER: RAGComponentMetadata(
                name="retriever", output_mapping={"retrieved_documents": "documents"}
            ),
            RAGComponent.RESPONSE_GENERATOR: RAGComponentMetadata(
                name="generator", output_mapping={"replies": "replies"}
            ),
        }

        return cls(rag_pipeline, rag_components, deepcopy(metrics))

    def _lookup_component_output(
        self, component: RAGComponent, outputs: Dict[str, Dict[str, Any]], output_name: str
    ) -> Any:
        name = self.rag_components[component].name
        mapping = self.rag_components[component].output_mapping
        output_name = mapping[output_name]
        return outputs[name][output_name]

    def _generate_eval_run_pipelines(self, overrides: Optional[RAGEvalOverrides]) -> PipelinePair:
        if overrides is None:
            overrides = RAGEvalOverrides()

        if overrides.eval is not None:
            overrides.eval = {k.value: v for k, v in overrides.eval.items()}  # type: ignore

        rag_pipeline = self._override_pipeline(self.rag_pipeline, overrides.rag)
        eval_pipeline = self._override_pipeline(self.evaluation_pipeline, overrides.eval)  # type: ignore

        return PipelinePair(
            first=rag_pipeline,
            second=eval_pipeline,
            outputs_to_inputs=self._map_rag_eval_pipeline_io(),
            map_first_outputs=lambda x: self._aggregate_rag_outputs(x),
            included_first_outputs={RAGComponent.DOCUMENT_RETRIEVER.value, RAGComponent.RESPONSE_GENERATOR.value},
        )

    def _aggregate_rag_outputs(self, outputs: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        aggregate = aggregate_batched_pipeline_outputs(outputs)

        # We only care about the first response from the generator.
        generator_name = self.rag_components[RAGComponent.RESPONSE_GENERATOR].name
        replies_output_name = self.rag_components[RAGComponent.RESPONSE_GENERATOR].output_mapping["replies"]
        aggregate[generator_name][replies_output_name] = [r[0] for r in aggregate[generator_name][replies_output_name]]

        return aggregate

    def _map_rag_eval_pipeline_io(self) -> Dict[str, List[str]]:
        # We currently only have metric components in the eval pipeline.
        # So, we just map those inputs to the outputs of the rag pipeline.
        metric_inputs_to_component_outputs = {
            RAGMetric.DOCUMENT_MAP: {"retrieved_documents": (RAGComponent.DOCUMENT_RETRIEVER, "retrieved_documents")},
            RAGMetric.DOCUMENT_MRR: {"retrieved_documents": (RAGComponent.DOCUMENT_RETRIEVER, "retrieved_documents")},
            RAGMetric.DOCUMENT_RECALL_SINGLE_HIT: {
                "retrieved_documents": (RAGComponent.DOCUMENT_RETRIEVER, "retrieved_documents")
            },
            RAGMetric.DOCUMENT_RECALL_MULTI_HIT: {
                "retrieved_documents": (RAGComponent.DOCUMENT_RETRIEVER, "retrieved_documents")
            },
            RAGMetric.SEMANTIC_ANSWER_SIMILARITY: {"predicted_answers": (RAGComponent.RESPONSE_GENERATOR, "replies")},
            RAGMetric.ANSWER_FAITHFULNESS: {
                "contexts": (RAGComponent.DOCUMENT_RETRIEVER, "retrieved_documents"),
                "responses": (RAGComponent.RESPONSE_GENERATOR, "replies"),
            },
        }

        outputs_to_inputs = {}
        for metric in self.metrics:
            io = metric_inputs_to_component_outputs[metric]
            for metric_input_name, (component, component_output_name) in io.items():
                component_out = f"{self.rag_components[component].name}.{self.rag_components[component].output_mapping[component_output_name]}"
                metric_in = f"{metric.value}.{metric_input_name}"
                if component_out not in outputs_to_inputs:
                    outputs_to_inputs[component_out] = []
                outputs_to_inputs[component_out].append(metric_in)

        return outputs_to_inputs

    def _prepare_rag_pipeline_inputs(self, inputs: RAGEvalInput) -> List[Dict[str, Dict[str, Any]]]:
        query_embedder_name = self.rag_components[RAGComponent.QUERY_PROCESSOR].name
        query_embedder_text_input = self.rag_components[RAGComponent.QUERY_PROCESSOR].input_mapping["query"]

        if inputs.additional_rag_inputs is not None:
            # Ensure that the query embedder input is not provided as additional input.
            existing = inputs.additional_rag_inputs.get(query_embedder_name)
            if existing is not None:
                existing = existing.get(query_embedder_text_input)
                if existing is not None:
                    raise ValueError(
                        f"Query embedder input '{query_embedder_text_input}' cannot be provided as additional input."
                    )

            # Add the queries as an aggregate input.
            rag_inputs = deepcopy(inputs.additional_rag_inputs)
            if query_embedder_name not in rag_inputs:
                rag_inputs[query_embedder_name] = {}
            rag_inputs[query_embedder_name][query_embedder_text_input] = deepcopy(inputs.queries)
        else:
            rag_inputs = {query_embedder_name: {query_embedder_text_input: deepcopy(inputs.queries)}}

        separate_rag_inputs = deaggregate_batched_pipeline_inputs(rag_inputs)
        return separate_rag_inputs

    def _prepare_eval_pipeline_additional_inputs(self, inputs: RAGEvalInput) -> Dict[str, Dict[str, Any]]:
        eval_inputs = {}

        for metric in self.metrics:
            if metric in (
                RAGMetric.DOCUMENT_MAP,
                RAGMetric.DOCUMENT_MRR,
                RAGMetric.DOCUMENT_RECALL_SINGLE_HIT,
                RAGMetric.DOCUMENT_RECALL_MULTI_HIT,
            ):
                if inputs.ground_truth_documents is None:
                    raise ValueError(f"Ground truth documents required for metric '{metric.value}'.")
                elif len(inputs.ground_truth_documents) != len(inputs.queries):
                    raise ValueError("Length of ground truth documents should match the number of queries.")

                eval_inputs[metric.value] = {"ground_truth_documents": inputs.ground_truth_documents}
            elif metric == RAGMetric.SEMANTIC_ANSWER_SIMILARITY:
                if inputs.ground_truth_answers is None:
                    raise ValueError(f"Ground truth answers required for metric '{metric.value}'.")
                elif len(inputs.ground_truth_answers) != len(inputs.queries):
                    raise ValueError("Length of ground truth answers should match the number of queries.")

                eval_inputs[metric.value] = {"ground_truth_answers": inputs.ground_truth_answers}
            elif metric == RAGMetric.ANSWER_FAITHFULNESS:
                eval_inputs[metric.value] = {"questions": inputs.queries}

        return eval_inputs

    def run(  # noqa: D102
        self,
        inputs: RAGEvalInput,
        *,
        overrides: Optional[RAGEvalOverrides] = None,
        run_name: Optional[str] = "RAG Evaluation",
    ) -> EvaluationRunResult:
        rag_inputs = self._prepare_rag_pipeline_inputs(inputs)
        eval_inputs = self._prepare_eval_pipeline_additional_inputs(inputs)
        pipeline_pair = self._generate_eval_run_pipelines(overrides)

        pipeline_outputs = pipeline_pair.run_first_as_batch(rag_inputs, eval_inputs)
        rag_outputs, eval_outputs = pipeline_outputs["first"], pipeline_outputs["second"]

        assert run_name is not None
        return EvaluationRunResult(
            run_name,
            inputs={
                "questions": inputs.queries,
                "contexts": [
                    [doc.content for doc in docs]
                    for docs in self._lookup_component_output(
                        RAGComponent.DOCUMENT_RETRIEVER, rag_outputs, "retrieved_documents"
                    )
                ],
                "responses": self._lookup_component_output(RAGComponent.RESPONSE_GENERATOR, rag_outputs, "replies"),
            },
            results=eval_outputs,
        )

    @staticmethod
    def _build_evaluation_pipeline(metrics: Set[RAGMetric]):
        pipeline = Pipeline()

        metric_ctors = {
            RAGMetric.DOCUMENT_MAP: DocumentMAPEvaluator,
            RAGMetric.DOCUMENT_MRR: DocumentMRREvaluator,
            RAGMetric.DOCUMENT_RECALL_SINGLE_HIT: partial(DocumentRecallEvaluator, mode=RecallMode.SINGLE_HIT),
            RAGMetric.DOCUMENT_RECALL_MULTI_HIT: partial(DocumentRecallEvaluator, mode=RecallMode.MULTI_HIT),
            RAGMetric.SEMANTIC_ANSWER_SIMILARITY: partial(SASEvaluator, model="sentence-transformers/all-MiniLM-L6-v2"),
            RAGMetric.ANSWER_FAITHFULNESS: FaithfulnessEvaluator,
        }

        for metric in metrics:
            ctor = metric_ctors[metric]
            pipeline.add_component(metric.value, ctor())

        return pipeline

    @staticmethod
    def _validate_rag_components(pipeline: Pipeline, components: Dict[RAGComponent, RAGComponentMetadata]):
        for e in RAGComponent:
            if e not in components:
                raise ValueError(f"RAG evaluation harness requires metadata for the '{e.value}' component.")

        pipeline_outputs = pipeline.outputs(include_components_with_connected_outputs=True)
        pipeline_inputs = pipeline.inputs(include_components_with_connected_inputs=True)

        for component, metadata in components.items():
            if metadata.name not in pipeline_outputs or metadata.name not in pipeline_inputs:
                raise ValueError(f"{component} named '{metadata.name}' not found in pipeline.")

            comp_inputs = pipeline_inputs[metadata.name]
            comp_outputs = pipeline_outputs[metadata.name]

            for needle in metadata.input_mapping.values():
                if needle not in comp_inputs:
                    raise ValueError(f"Required input '{needle}' not found in component '{metadata.name}'.")

            for needle in metadata.output_mapping.values():
                if needle not in comp_outputs:
                    raise ValueError(f"Required output '{needle}' not found in component '{metadata.name}'.")
