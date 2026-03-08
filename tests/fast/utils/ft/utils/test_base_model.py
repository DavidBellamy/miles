"""Tests for miles.utils.ft.utils.base_model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.ft.utils.base_model import FtBaseModel


class _SampleModel(FtBaseModel):
    name: str
    value: int = 0


class TestFtBaseModel:
    def test_subclass_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            _SampleModel(name="test", value=1, unknown_field="oops")

    def test_subclass_accepts_known_fields(self) -> None:
        model = _SampleModel(name="hello", value=42)
        assert model.name == "hello"
        assert model.value == 42

    def test_extra_forbid_config(self) -> None:
        assert FtBaseModel.model_config.get("extra") == "forbid"
