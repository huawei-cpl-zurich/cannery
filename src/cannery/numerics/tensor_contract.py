"""Frozen tensor case domains; independent of native loading and author code."""

import hashlib
import json
import math
from typing import Literal, Self

from pydantic import Field, model_validator

from cannery.numerics.baseline_v2 import OutputContractV2, OutputTensorV2
from cannery.numerics.recipes import MAX_SUITE_BYTES
from cannery.numerics.recipes_v2 import RecipeSuiteV2
from cannery.registry.manifest import Name, StrictModel, parse_yaml_mapping


class ScalarValueV2(StrictModel):
    kind: Literal["bool", "int64", "float64"]
    value: bool | int | float

    @model_validator(mode="after")
    def typed_value(self) -> Self:
        valid = (
            self.kind == "bool"
            and type(self.value) is bool
            or self.kind == "int64"
            and type(self.value) is int
            and -(2**63) <= self.value < 2**63
            or self.kind == "float64"
            and type(self.value) is float
            and math.isfinite(self.value)
        )
        if not valid:
            raise ValueError("Invalid typed scalar value")
        return self


class TensorCaseV2(StrictModel):
    outputs: dict[Name, OutputTensorV2] = Field(min_length=1, max_length=32)
    scalars: dict[Name, ScalarValueV2] = Field(default_factory=dict, max_length=32)
    aliases: dict[Name, Name] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def output_aliases(self) -> Self:
        if not set(self.aliases) <= set(self.outputs):
            raise ValueError("Alias must name a declared output")
        if len(set(self.aliases.values())) != len(self.aliases):
            raise ValueError("Each mutable input has exactly one returned alias")
        return self


class TensorContractV2(StrictModel):
    schema_version: Literal[2]
    cases: dict[Name, TensorCaseV2] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def bounded_outputs(self) -> Self:
        self.output_contract()
        return self

    def output_contract(self) -> OutputContractV2:
        return OutputContractV2(cases={name: case.outputs for name, case in self.cases.items()})

    def validate_suite(self, suite: RecipeSuiteV2) -> None:
        if set(self.cases) != {case.id for case in suite.cases}:
            raise ValueError("Tensor contract must cover every recipe case exactly once")
        interface = None
        dispatch: dict[str, str] = {}
        for recipe_case in suite.cases:
            case = self.cases[recipe_case.id]
            if set(recipe_case.inputs) & set(case.scalars):
                raise ValueError("Tensor and scalar argument names must be distinct")
            for recipe in recipe_case.inputs.values():
                if recipe.layout != "contiguous":
                    raise ValueError("Tensor-v2 requires contiguous input recipes")
            for output, input_name in case.aliases.items():
                input_recipe = recipe_case.inputs.get(input_name)
                descriptor = case.outputs[output]
                if input_recipe is None or (input_recipe.shape, input_recipe.dtype) != (
                    descriptor.shape,
                    descriptor.dtype,
                ):
                    raise ValueError("Mutable alias must match an input shape and dtype")
            size = sum(r.nbytes for r in recipe_case.inputs.values()) + sum(
                out.nbytes for out in case.outputs.values()
            )
            if size > MAX_SUITE_BYTES:
                raise ValueError("Case exceeds native tensor byte budget")
            current = (
                tuple(sorted(recipe_case.inputs)),
                tuple(sorted(case.outputs)),
                tuple(sorted((name, s.kind) for name, s in case.scalars.items())),
            )
            if interface is not None and interface != current:
                raise ValueError("Tensor cases must share a named interface")
            interface = current
            domain = json.dumps(
                {
                    "inputs": {
                        name: {"shape": r.shape, "dtype": r.dtype}
                        for name, r in recipe_case.inputs.items()
                    },
                    "scalars": {name: s.model_dump() for name, s in case.scalars.items()},
                },
                sort_keys=True,
            )
            output_signature = json.dumps(
                {
                    "outputs": {
                        name: out.model_dump(mode="json") for name, out in case.outputs.items()
                    },
                    "aliases": case.aliases,
                },
                sort_keys=True,
            )
            if domain in dispatch and dispatch[domain] != output_signature:
                raise ValueError("Ambiguous native output domain")
            dispatch[domain] = output_signature

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def parse_tensor_contract(data: bytes, suite: RecipeSuiteV2) -> TensorContractV2:
    contract = TensorContractV2.model_validate(parse_yaml_mapping(data))
    contract.validate_suite(suite)
    return contract
