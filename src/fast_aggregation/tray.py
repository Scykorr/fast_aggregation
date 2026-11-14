"""Иконка в трее Windows: статус, дашборд, журнал, перезапуск.

pystray и Pillow импортируются только здесь, чтобы команда scan работала
без них. В CI нет сеанса рабочего стола, поэтому трей там не запускается.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import webbrowser
from collections.abc import Callable


def run_tray(
    *,
    url_getter: Callable[[], str | None] | None,
    status: Callable[[], str] | None = None,
    logs_dir: str | None = None,
    on_restart: Callable[[], None] | None = None,
    on_quit: Callable[[], None] | None = None,
) -> None:
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit(
            "Для иконки в трее нужны пакеты pystray и pillow. "
            "Их ставит сборка Windows. Ошибка: "
            + str(exc)
        ) from exc

    image = Image.new("RGB", (64, 64), (16, 20, 15))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 56, 56), fill=(198, 242, 122))
    title = {"text": "fastagg"}

    def open_dashboard(_icon: object, _item: object) -> None:
        if url_getter is None:
            return
        url = url_getter()
        if url:
            webbrowser.open(url)

    def show_logs(_icon: object, _item: object) -> None:
        if not logs_dir:
            return
        try:
            if sys.platform == "win32":
                os.startfile(logs_dir)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", logs_dir])
        except OSError:
            return

    def restart(_icon: object, _item: object) -> None:
        if on_restart is not None:
            on_restart()

    def quit_app(icon: object, _item: object) -> None:
        if on_quit is not None:
            on_quit()
        icon.stop()
        os._exit(0)

    def refresh(icon: object) -> None:
        while True:
            if status is not None:
                try:
                    title["text"] = "fastagg: " + status()
                except Exception:
                    title["text"] = "fastagg: нет ответа"
                icon.title = title["text"]
            threading.Event().wait(3)

    menu_items = []
    if url_getter is not None:
        menu_items.append(pystray.MenuItem("Открыть дашборд", open_dashboard, default=True))
    if logs_dir:
        menu_items.append(pystray.MenuItem("Показать журналы", show_logs))
    if on_restart is not None:
        menu_items.append(pystray.MenuItem("Перезапустить", restart))
    menu_items.append(pystray.MenuItem("Выход", quit_app))
    icon = pystray.Icon("fastagg", image, "fastagg", pystray.Menu(*menu_items))
    threading.Thread(target=refresh, args=(icon,), daemon=True).start()
    icon.run()
