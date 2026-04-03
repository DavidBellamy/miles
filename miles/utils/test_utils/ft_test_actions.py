import json
import logging
from typing import Literal

from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)


class FTTestAction(FrozenStrictBaseModel):
    after_step: int
    action: Literal["stop_cell", "start_cell"]
    cell_index: int  # -1 = last cell


def parse_ft_test_actions(raw: str | None) -> list[FTTestAction]:
    if not raw:
        return []
    return [FTTestAction(**item) for item in json.loads(raw)]
