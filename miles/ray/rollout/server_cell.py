from dataclasses import dataclass


@dataclass(frozen=True)
class ServerCellConfig:
    TODO


class ServerCell:
    def __init__(self, config: ServerCellConfig):
        self.config = config
        TODO
