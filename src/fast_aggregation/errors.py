class FastAggError(Exception):
    """Ошибка, которую можно показать пользователю в API и CLI."""


class PlacementError(FastAggError):
    """Кластер нельзя разложить по устройствам с текущими правилами."""


class BackendError(FastAggError):
    """Запрошенный бэкенд llama.cpp на узле недоступен."""


class RuntimeError_(FastAggError):
    """Не удалось скачать или запустить бинарник llama.cpp.

    Имя с подчёркиванием, чтобы не затенять встроенный RuntimeError
    в модулях, которые импортируют оба.
    """
