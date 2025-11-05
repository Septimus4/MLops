from __future__ import annotations

from api.model_loader import ModelMetadata, get_model


def test_model_loads_and_has_metadata() -> None:
    model, metadata = get_model()
    assert hasattr(model, "predict")
    assert isinstance(metadata, ModelMetadata)
    assert metadata.model_name
    assert metadata.model_version
    assert metadata.feature_count > 0
