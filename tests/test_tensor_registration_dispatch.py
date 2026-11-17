import pytest
from test_recipe_spec import recipe_spec
from test_tensor_spec_reader import tensor_files

from cannery.registry.intake import registration_spec
from cannery.registry.manifest import SpecError, validate_spec
from cannery.registry.tensor_manifest import validate_tensor_spec


def test_registration_dispatch_requires_explicit_operator_opt_in():
    files = tensor_files()
    with pytest.raises(SpecError):
        registration_spec(files, "spec")
    assert (
        registration_spec(files, "spec", tensor_enabled=True)
        == validate_tensor_spec(files, "spec").spec
    )
    # This fixture's reference raises at import; no author imports occur.
    root = {name.removeprefix("spec/"): value for name, value in files.items()}
    assert registration_spec(root, ".", tensor_enabled=True) == validate_tensor_spec(root, ".").spec


def test_unary_registration_identity_unchanged_with_either_flag():
    files, _, _ = recipe_spec()
    for enabled in [False, True]:
        assert registration_spec(files, "spec", tensor_enabled=enabled) == validate_spec(
            files, "spec"
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_missing_manifest_rejected_as_invalid_spec(enabled):
    with pytest.raises(SpecError):
        registration_spec({}, "spec", tensor_enabled=enabled)
