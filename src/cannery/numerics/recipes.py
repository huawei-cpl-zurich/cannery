"""Versioned portable float32 recipes; no author Python or framework RNG."""

import hashlib
import json
import math
import struct
from typing import Annotated, Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from cannery.registry.manifest import Name, StrictModel, parse_yaml_mapping

GENERATOR = "sha256-f32-v1"
MAX_TENSOR_ELEMENTS = 1048576
MAX_SUITE_BYTES = 32 * 1024 * 1024
Dimension = Annotated[int, Field(ge=0, le=MAX_TENSOR_ELEMENTS)]


class TensorRecipe(StrictModel):
    shape: tuple[Dimension, ...] = Field(min_length=1, max_length=4)
    dtype: Literal["float32"] = "float32"
    distribution: Literal["uniform_f32", "arange_f32", "zeros_f32"]
    seed: int = Field(ge=0, le=2**64 - 1)
    layout: Literal["contiguous", "transpose"] = "contiguous"

    @field_validator("shape", mode="before")
    @classmethod
    def dimensions(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def bounded_tensor(self) -> Self:
        if math.prod(self.shape) > MAX_TENSOR_ELEMENTS:
            raise ValueError("Tensor exceeds element budget")
        if self.layout == "transpose" and len(self.shape) != 2:
            raise ValueError("Transpose layout requires exactly two dimensions")
        return self


class RecipeCase(StrictModel):
    id: Name
    inputs: dict[Name, TensorRecipe] = Field(min_length=1, max_length=1)


class RecipeSuite(StrictModel):
    schema_version: Literal[1]
    generator: Literal["sha256-f32-v1"]
    cases: tuple[RecipeCase, ...] = Field(min_length=1, max_length=32)

    @field_validator("cases", mode="before")
    @classmethod
    def case_list(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def bounded_suite(self) -> Self:
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Duplicate case identifier")
        total = sum(4 * math.prod(r.shape) for c in self.cases for r in c.inputs.values())
        if total > MAX_SUITE_BYTES:
            raise ValueError("Suite exceeds aggregate input budget")
        return self

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def parse_recipes(data: bytes) -> RecipeSuite:
    return RecipeSuite.model_validate(parse_yaml_mapping(data))


def generate(recipe: TensorRecipe, case_id: str, input_name: str) -> bytes:
    """Return little-endian float32 values in logical contiguous element order.

    Transpose recipes fill a reversed-shape backing tensor, then transpose it.
    Consumers must reconstruct that view; bytes here describe logical values.
    """
    # Validate identifiers even for direct callers; NUL is a framing delimiter.
    RecipeCase.model_validate({"id": case_id, "inputs": {input_name: recipe}})
    count = math.prod(recipe.shape)
    if recipe.distribution == "zeros_f32":
        data = bytes(4 * count)
    elif recipe.distribution == "arange_f32":
        data = b"".join(struct.pack("<f", i) for i in range(count))
    else:
        prefix = (
            b"cannery-input-v1\0"
            + struct.pack("<Q", recipe.seed)
            + case_id.encode("utf-8")
            + b"\0"
            + input_name.encode("utf-8")
            + b"\0"
        )
        output = bytearray()
        for block in range((count + 7) // 8):
            digest = hashlib.sha256(prefix + struct.pack("<Q", block)).digest()
            for word in struct.unpack("<8I", digest):
                value = ((word >> 8) - 2**23) * 2**-23
                output.extend(struct.pack("<f", value))
        data = bytes(output[: count * 4])
    if recipe.layout == "transpose":
        rows, columns = recipe.shape
        data = b"".join(
            data[(column * rows + row) * 4 : (column * rows + row + 1) * 4]
            for row in range(rows)
            for column in range(columns)
        )
    return data


def input_sha256(recipe: TensorRecipe, data: bytes) -> str:
    if len(data) != math.prod(recipe.shape) * 4:
        raise ValueError("Input byte length does not match recipe")
    descriptor = json.dumps(
        {"shape": recipe.shape, "dtype": recipe.dtype, "layout": recipe.layout},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(descriptor + b"\0" + data).hexdigest()
