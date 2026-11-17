"""Bind source schema to the immutable specification and operator rollout gate."""

from cannery.submissions.manifest import (
    SubmissionError,
    ValidatedSubmission,
    validate_submission,
    validate_tensor_submission,
)


def submission_for_spec(
    files: dict[str, bytes], source_path: str, schema: int, *, tensor_enabled: bool = False
) -> ValidatedSubmission:
    if schema == 2 and tensor_enabled:
        return validate_tensor_submission(files, source_path)
    if schema != 1:
        raise SubmissionError("Tensor submission intake is not enabled")
    return validate_submission(files, source_path)
