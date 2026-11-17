"""Feature-gated dispatch for registration; both readers remain source-only."""

from cannery.git_intake import relative_source_path
from cannery.registry.manifest import SpecError, ValidatedSpec, parse_manifest, validate_spec
from cannery.registry.tensor_manifest import validate_tensor_spec


def registration_spec(
    files: dict[str, bytes], source_path: str, *, tensor_enabled: bool = False
) -> ValidatedSpec:
    prefix = "" if source_path == "." else relative_source_path(source_path) + "/"
    try:
        if tensor_enabled and parse_manifest(files[prefix + "cannery.yaml"]).schema_version == 2:
            return validate_tensor_spec(files, source_path).spec
    except KeyError:
        raise SpecError("Specification manifest is missing") from None
    return validate_spec(files, source_path)
