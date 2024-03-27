from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from haystack import Pipeline


@dataclass(frozen=True)
class PipelinePair:
    """
    A pair of pipelines that are linked together and
    executed sequentially.

    :param first:
        The first pipeline in the sequence.
    :param second:
        The second pipeline in the sequence.
    :param outputs_to_inputs:
        A mapping of the outputs of the first pipeline to the
        inputs of the second pipeline in the following format:
        `"name_of_component.name_of_output": "name_of_component.name_of_input`.
        A single output can be mapped to multiple inputs.
    :param map_first_outputs:
        A function that post-processes the outputs of the first
        pipeline, which it receives as its (only) argument.
    :param included_first_outputs:
        Names of components in the first pipeline whose outputs
        should be included in the final outputs.
    :param included_second_outputs:
        Names of components in the second pipeline whose outputs
        should be included in the final outputs.
    """

    first: Pipeline
    second: Pipeline
    outputs_to_inputs: Dict[str, List[str]]
    map_first_outputs: Optional[Callable] = None
    included_first_outputs: Optional[Set[str]] = None
    included_second_outputs: Optional[Set[str]] = None

    def __post_init__(self):
        first_outputs = self.first.outputs(include_components_with_connected_outputs=True)
        second_inputs = self.second.inputs(include_components_with_connected_inputs=True)
        seen_second_inputs = set()

        # Validate the mapping of outputs from the first pipeline
        # to the inputs of the second pipeline.
        for first_out, second_ins in self.outputs_to_inputs.items():
            first_comp_name, first_out_name = self._split_input_output_path(first_out)
            if first_comp_name not in first_outputs:
                raise ValueError(f"Output component '{first_comp_name}' not found in first pipeline.")
            elif first_out_name not in first_outputs[first_comp_name]:
                raise ValueError(
                    f"Component '{first_comp_name}' in first pipeline does not have expected output '{first_out_name}'."
                )

            for second_in in second_ins:
                if second_in in seen_second_inputs:
                    raise ValueError(
                        f"Input '{second_in}' in second pipeline is connected to multiple first pipeline outputs."
                    )

                second_comp_name, second_input_name = self._split_input_output_path(second_in)
                if second_comp_name not in second_inputs:
                    raise ValueError(f"Input component '{second_comp_name}' not found in second pipeline.")
                elif second_input_name not in second_inputs[second_comp_name]:
                    raise ValueError(
                        f"Component '{second_comp_name}' in second pipeline does not have expected input '{second_input_name}'."
                    )
                seen_second_inputs.add(second_in)

    def _validate_second_inputs(self, inputs: Dict[str, Dict[str, Any]]):
        # Check if the connected input is also provided explicitly.
        second_connected_inputs = [
            self._split_input_output_path(p_h) for p in self.outputs_to_inputs.values() for p_h in p
        ]
        for component_name, input_name in second_connected_inputs:
            provided_input = inputs.get(component_name)
            if provided_input is None:
                continue
            elif input_name in provided_input:
                raise ValueError(
                    f"Second pipeline input '{component_name}.{input_name}' cannot be provided both explicitly and by the first pipeline."
                )

    def _split_input_output_path(self, path: str) -> Tuple[str, str]:
        # Split the input/output path into component name and input/output name.
        pos = path.find(".")
        if pos == -1:
            raise ValueError(
                f"Invalid pipeline i/o path specifier '{path}' - Must be in the following format: <component_name>.<input/output_name>"
            )
        return path[:pos], path[pos + 1 :]

    def _prepare_reqd_outputs_for_first_pipeline(self) -> Set[str]:
        # To ensure that we have all the outputs from the first
        # pipeline that are required by the second pipeline.
        first_components_with_outputs = {self._split_input_output_path(p)[0] for p in self.outputs_to_inputs.keys()}
        if self.included_first_outputs is not None:
            first_components_with_outputs = first_components_with_outputs.union(self.included_first_outputs)
        return first_components_with_outputs

    def _map_first_second_pipeline_io(
        self, first_outputs: Dict[str, Dict[str, Any]], second_inputs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        # Map the first pipeline outputs to the second pipeline inputs.
        for first_output, second_input_candidates in self.outputs_to_inputs.items():
            first_component, first_output = self._split_input_output_path(first_output)

            # Each output from the first pipeline can be mapped to multiple inputs in the second pipeline.
            for second_input in second_input_candidates:
                second_component, second_input_socket = self._split_input_output_path(second_input)

                second_component_inputs = second_inputs.get(second_component)
                if second_component_inputs is not None:
                    # Pre-condition should've been validated earlier.
                    assert second_input_socket not in second_component_inputs
                    # The first pipeline's output should also guaranteed at this point.
                    second_component_inputs[second_input_socket] = first_outputs[first_component][first_output]
                else:
                    second_inputs[second_component] = {
                        second_input_socket: first_outputs[first_component][first_output]
                    }

        return second_inputs

    def run(
        self, first_inputs: Dict[str, Dict[str, Any]], second_inputs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute the pipeline pair by invoking first pipeline
        and then the second with the outputs of the former. This
        assumes that both pipelines have the same input modality,
        i.e., the shapes of the first pipeline's outputs match the
        shapes of the second pipeline's inputs.

        :param first_inputs:
            The inputs to the first pipeline.
        :param second_inputs:
            The inputs to the second pipeline.
        :returns:
            A dictionary with the following keys:
            - `first` - The outputs of the first pipeline.
            - `second` - The outputs of the second pipeline.
        """
        second_inputs = second_inputs or {}
        self._validate_second_inputs(second_inputs)

        first_outputs = self.first.run(
            first_inputs, include_outputs_from=self._prepare_reqd_outputs_for_first_pipeline()
        )
        if self.map_first_outputs is not None:
            first_outputs = self.map_first_outputs(first_outputs)
        second_inputs = self._map_first_second_pipeline_io(first_outputs, second_inputs)
        second_outputs = self.second.run(second_inputs, include_outputs_from=self.included_second_outputs)

        return {"first": first_outputs, "second": second_outputs}

    def run_first_as_batch(
        self, first_inputs: List[Dict[str, Dict[str, Any]]], second_inputs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute the pipeline pair by invoking the first pipeline
        iteratively over the list of inputs and passing the cumulative
        outputs to the second pipeline. This is suitable when the first
        pipeline has a single logical input-to-output mapping and the
        second pipeline expects multiple logical inputs, e.g: a retrieval
        pipeline that accepts a single query and returns a list of documents
        and an evaluation pipeline that accepts multiple lists of documents
        and multiple lists of ground truth data.

        :param first_inputs:
            A batch of inputs to the first pipeline. A mapping
            function must be provided to aggregate the outputs.
        :param second_inputs:
            The inputs to the second pipeline.
        :returns:
            A dictionary with the following keys:
            - `first` - The (aggregate) outputs of the first pipeline.
            - `second` - The outputs of the second pipeline.
        """
        second_inputs = second_inputs or {}
        self._validate_second_inputs(second_inputs)

        first_components_with_outputs = self._prepare_reqd_outputs_for_first_pipeline()
        if self.map_first_outputs is None:
            raise ValueError("Mapping function for first pipeline outputs must be provided for batch execution.")

        first_outputs: Dict[str, Dict[str, Any]] = self.map_first_outputs(
            [self.first.run(i, include_outputs_from=first_components_with_outputs) for i in first_inputs]
        )
        if not isinstance(first_outputs, dict):
            raise ValueError("Mapping function must return an aggregate dictionary of outputs.")

        second_inputs = self._map_first_second_pipeline_io(first_outputs, second_inputs)
        second_outputs = self.second.run(second_inputs, include_outputs_from=self.included_second_outputs)

        return {"first": first_outputs, "second": second_outputs}


def aggregate_batched_pipeline_outputs(outputs: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Combine the outputs of a pipeline that has been executed
    iteratively over a batch of inputs. It performs a transpose
    operation on the first and the third dimensions of the outputs.

    :param outputs:
        A list of outputs from the pipeline, where each output
        is a dictionary with the same keys and values with the
        same keys.
    :returns:
        The combined outputs.
    """
    # The pipeline is invoked iteratively over a batch of inputs, such
    # that each element in the outputs corresponds to a single element in
    # the batch input.
    if len(outputs) == 0:
        return {}
    elif len(outputs) == 1:
        return outputs[0]

    # We'll use the first output as a sentinel to determine
    # if the shape of the rest of the outputs are the same.
    sentinel = outputs[0]
    for output in outputs[1:]:
        if output.keys() != sentinel.keys():
            raise ValueError(
                f"Expected outputs from components '{list(sentinel.keys())}' but got components '{list(output.keys())}'"
            )

        for component_name, expected in sentinel.items():
            got = output[component_name]
            if got.keys() != expected.keys():
                raise ValueError(
                    f"Expected outputs from component '{component_name}' to have keys '{list(expected.keys())}' but got '{list(got.keys())}'"
                )

    # The outputs are of the correct/same shape. Now to transpose
    # the outermost list with the innermost dictionary.
    transposed: Dict[str, Dict[str, Any]] = {}
    for k, v in sentinel.items():
        transposed[k] = {k_h: [] for k_h in v.keys()}

    for output in outputs:
        for component_name, component_outputs in output.items():
            dest = transposed[component_name]
            for output_name, output_value in component_outputs.items():
                dest[output_name].append(output_value)

    return transposed


def deaggregate_batched_pipeline_inputs(inputs: Dict[str, Dict[str, List[Any]]]) -> List[Dict[str, Dict[str, Any]]]:
    """
    Separate the inputs of a pipeline that has been batched along
    its innermost dimension (component -> input -> values). It
    performs a transpose operation on the first and the third dimensions
    of the inputs.

    :param inputs:
        A dictionary of pipeline inputs that maps
        component-input pairs to lists of values.
    :returns:
        The separated inputs.
    """
    if len(inputs) == 0:
        return []

    sentinel = next(iter(inputs.values()))  # First component's inputs
    sentinel = next(iter(sentinel.values()))  # First component's first input's values

    for component_name, component_inputs in inputs.items():
        for input_name, input_values in component_inputs.items():
            if len(input_values) != len(sentinel):
                raise ValueError(
                    f"Expected input '{component_name}.{input_name}' to have {len(sentinel)} values but got {len(input_values)}"
                )

    proto = {k: {k_h: None for k_h in v.keys()} for k, v in inputs.items()}
    transposed: List[Dict[str, Dict[str, Any]]] = []

    for i in range(len(sentinel)):
        new_dict = deepcopy(proto)
        for component_name, component_inputs in inputs.items():
            for input_name, input_values in component_inputs.items():
                new_dict[component_name][input_name] = input_values[i]
        transposed.append(new_dict)

    return transposed
