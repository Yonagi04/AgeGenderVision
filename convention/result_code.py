class ResultCode:
    SUCCESS = 200           # 成功
    NO_DATA = 204           # 成功，但无数据（空列表、空对象）
    NOT_FOUND = 404         # 失败，且请求的资源不存在（搜索无结果）
    PANEL_ERROR = 400      # 失败，Panel层请求参数错误
    SERVER_ERROR = 500      # 失败，服务层异常
