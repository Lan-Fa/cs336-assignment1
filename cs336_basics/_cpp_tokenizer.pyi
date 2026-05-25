class CppBPE:
    def __init__(
        self,
        token_to_id: dict[bytes, int],
        merges: list[tuple[bytes, bytes]],
    ) -> None: ...

    def encode_bytes(self, input_bytes: bytes) -> list[int]: ...