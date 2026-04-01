from typing import Annotated, Literal, Union

from pydantic import Discriminator

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class _ProcessIdentityBase(FrozenStrictBaseModel):
    def to_name(self) -> str:
        raise NotImplementedError


class MainProcessIdentity(_ProcessIdentityBase):
    component: Literal["main"] = "main"

    def to_name(self) -> str:
        return "main"


class RolloutManagerProcessIdentity(_ProcessIdentityBase):
    component: Literal["rollout_manager"] = "rollout_manager"

    def to_name(self) -> str:
        return "rollout_manager"


class TrainProcessIdentity(_ProcessIdentityBase):
    component: Literal["train"] = "train"
    cell_index: int
    rank_within_cell: int

    def to_name(self) -> str:
        return f"train_cell{self.cell_index}_rank{self.rank_within_cell}"


ProcessIdentity = Annotated[
    Union[MainProcessIdentity, RolloutManagerProcessIdentity, TrainProcessIdentity],
    Discriminator("component"),
]
