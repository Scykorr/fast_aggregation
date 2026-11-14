"""Иконка в трее Windows: статус и открытие консоли.

pystray и Pillow импортируются только здесь, чтобы команда scan работала
без них. В CI нет сеанса рабочего стола, поэтому трей там не запускается.
"""

from __future__ import annotations

import os
import threading
import webbrowser


def run_tray(console_url: str | None, status: callable | None = None) -> None:
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

    def open_console(_icon: object, _item: object) -> None:
        if console_url:
            webbrowser.open(console_url)

    def quit_app(icon: object, _item: object) -> None:
        icon.stop()
        os._exit(0)

    def refresh(icon: object) -> None:
        while True:
            if status is not None:
                try:
                    title["text"] = "fastagg: " + status()
                except Exception:
                    title["text"] = "fastagg"
                icon.title = title["text"]
            threading.Event().wait(3)

    menu_items = []
    if console_url:
        menu_items.append(pystray.MenuItem("Открыть консоль", open_console, default=True))
    menu_items.append(pystray.MenuItem("Выход", quit_app))
    icon = pystray.Icon("fastagg", image, "fastagg", pystray.Menu(*menu_items))
    threading.Thread(target=refresh, args=(icon,), daemon=True).start()
    icon.run()
