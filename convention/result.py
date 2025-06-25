from typing import Generic, TypeVar, Optional

T = TypeVar("T")

class Result(Generic[T]):
    def __init__(self, success: bool, data: Optional[T] = None, message: str = "", code: int = 200):
        self.success = success
        self.data = data
        self.message = message
        self.code = code

    @staticmethod
    def success(data: Optional[T] = None, message: str = "操作成功", code: int = 200):
        return Result(True, data, message, code)

    @staticmethod
    def fail(message: str = "操作失败", data: Optional[T] = None, code: int = 500):
        return Result(False, data, message, code)

    def __repr__(self):
        return f"<Result success={self.success}, message={self.message}, data={self.data}, code={self.code}>"
