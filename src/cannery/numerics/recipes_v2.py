"""Portable mixed-dtype recipes. Not admitted by the v1 registration path.

sha256-tensor-v2 uses framed, independent uint32 streams. Float samples use
the upper 24 bits mapped to [-1, 1), rounded to the requested storage dtype.
Integer samples use modulo reduction into an inclusive interval (not an
unbiased statistical RNG). Permutations use descending Fisher-Yates swaps,
one uint32 modulo (index + 1) per swap from the same framed stream. They
contain every index in [0, count) exactly once. All storage is little endian,
in logical order. Sequence recipes use evenly spaced inclusive integer endpoints
with an exact integral step, independent of RNG.
Existing sha256-f32-v1 recipes and their hashes remain unchanged.
"""

import hashlib
import json
import math
import struct
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from cannery.numerics.recipes import MAX_SUITE_BYTES, MAX_TENSOR_ELEMENTS, Dimension
from cannery.registry.manifest import Name, StrictModel, parse_yaml_mapping

GENERATOR = "sha256-tensor-v2"
DType = Literal["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]
ITEM_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}
INTEGER_LIMITS = {
    "int8": (-128, 127),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
    "bool": (0, 1),
}


class TensorRecipeV2(StrictModel):
    shape: tuple[Dimension, ...] = Field(min_length=1, max_length=4)
    dtype: DType
    distribution: Literal["uniform", "randint", "zeros", "permutation", "sequence"]
    seed: int = Field(ge=0, le=2**64 - 1)
    low: int | None = None
    high: int | None = None
    layout: Literal["contiguous", "transpose"] = "contiguous"

    @field_validator("shape", mode="before")
    @classmethod
    def dimensions(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if math.prod(self.shape) > MAX_TENSOR_ELEMENTS:
            raise ValueError("Tensor exceeds element budget")
        if self.layout == "transpose" and len(self.shape) != 2:
            raise ValueError("Transpose requires two dimensions")
        if self.distribution == "uniform" and self.dtype in INTEGER_LIMITS:
            raise ValueError("Uniform requires a floating dtype")
        if self.distribution in {"permutation", "sequence"} and (
            self.dtype not in {"int32", "int64"} or len(self.shape) != 1
        ):
            raise ValueError("Structured indices require a one-dimensional int32/int64 tensor")
        if self.distribution in {"randint", "sequence"}:
            if self.dtype not in INTEGER_LIMITS or self.low is None or self.high is None:
                raise ValueError("Randint requires an integer dtype and bounds")
            minimum, maximum = INTEGER_LIMITS[self.dtype]
            if not minimum <= self.low <= self.high <= maximum:
                raise ValueError("Integer bounds exceed dtype range")
            if self.distribution == "randint" and self.high - self.low >= 2**32:
                raise ValueError("Integer interval exceeds uint32 stream range")
            count = math.prod(self.shape)
            if self.distribution == "sequence" and (
                (count <= 1 and self.high != self.low)
                or (count > 1 and (self.high - self.low) % (count - 1))
            ):
                raise ValueError("Sequence endpoints must define an integral step")
        elif self.low is not None or self.high is not None:
            raise ValueError("Only randint and sequence accept bounds")
        return self

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * ITEM_BYTES[self.dtype]


class RecipeCaseV2(StrictModel):
    id: Name
    inputs: dict[Name, TensorRecipeV2] = Field(min_length=1, max_length=32)


class RecipeSuiteV2(StrictModel):
    schema_version: Literal[2]
    generator: Literal["sha256-tensor-v2"]
    cases: tuple[RecipeCaseV2, ...] = Field(min_length=1, max_length=32)

    @field_validator("cases", mode="before")
    @classmethod
    def case_list(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Duplicate case identifier")
        if sum(r.nbytes for c in self.cases for r in c.inputs.values()) > MAX_SUITE_BYTES:
            raise ValueError("Suite exceeds aggregate input budget")
        return self

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def parse_recipes_v2(data: bytes) -> RecipeSuiteV2:
    return RecipeSuiteV2.model_validate(parse_yaml_mapping(data))


def _float_bytes(value: float, dtype: DType) -> bytes:
    if dtype == "float16":
        return struct.pack("<e", value)
    if dtype == "bfloat16":
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        # Round-to-nearest, ties-to-even. Generated values are always finite.
        return struct.pack("<H", (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16)
    return struct.pack("<f", value)


def generate_v2(recipe: TensorRecipeV2, case_id: str, input_name: str) -> bytes:
    """Generate bytes without framework RNG or executing author code.

    For transpose, fill reversed-shape backing storage then transpose; callers
    must reconstruct that view. The returned bytes are in logical order.
    """
    RecipeCaseV2.model_validate({"id": case_id, "inputs": {input_name: recipe}})
    return _generate_v2(recipe, case_id, input_name)


def _generate_v2(recipe: TensorRecipeV2, case_id: str, input_name: str) -> bytes:
    """Pure stdlib generation for isolated trusted supervisors after validation."""
    count = math.prod(recipe.shape)
    width = ITEM_BYTES[recipe.dtype]
    if recipe.distribution == "zeros":
        return bytes(count * width)
    if recipe.distribution == "sequence":
        assert recipe.low is not None and recipe.high is not None
        step = (recipe.high - recipe.low) // (count - 1) if count > 1 else 0
        return b"".join(
            (recipe.low + index * step).to_bytes(width, "little", signed=True)
            for index in range(count)
        )
    prefix = (
        b"cannery-input-v2\0"
        + struct.pack("<Q", recipe.seed)
        + case_id.encode()
        + b"\0"
        + input_name.encode()
        + b"\0"
    )
    if recipe.distribution == "permutation":
        values = list(range(count))
        index = count - 1
        for block in range((max(0, count - 1) + 7) // 8):
            words = struct.unpack("<8I", hashlib.sha256(prefix + struct.pack("<Q", block)).digest())
            for word in words:
                if index <= 0:
                    break
                other = word % (index + 1)
                values[index], values[other] = values[other], values[index]
                index -= 1
        return b"".join(value.to_bytes(width, "little", signed=True) for value in values)
    output = bytearray()
    for block in range((count + 7) // 8):
        words = struct.unpack("<8I", hashlib.sha256(prefix + struct.pack("<Q", block)).digest())
        for word in words[: min(8, count - block * 8)]:
            if recipe.distribution == "randint":
                assert recipe.low is not None and recipe.high is not None
                value = recipe.low + word % (recipe.high - recipe.low + 1)
                output.extend(value.to_bytes(width, "little", signed=recipe.dtype != "bool"))
            else:
                output.extend(_float_bytes(((word >> 8) - 2**23) * 2**-23, recipe.dtype))
    data = bytes(output)
    if recipe.layout == "transpose":
        rows, columns = recipe.shape
        data = b"".join(
            data[(column * rows + row) * width : (column * rows + row + 1) * width]
            for row in range(rows)
            for column in range(columns)
        )
    return data


def input_sha256_v2(recipe: TensorRecipeV2, data: bytes) -> str:
    if len(data) != recipe.nbytes:
        raise ValueError("Input byte length does not match recipe")
    descriptor = json.dumps(
        {"shape": recipe.shape, "dtype": recipe.dtype, "layout": recipe.layout},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(descriptor + b"\0" + data).hexdigest()
