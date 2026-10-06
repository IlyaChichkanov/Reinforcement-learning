"""Встроить картинки и анимации в ноутбук вложениями (attachments).

Ноутбук ссылается на медиа как ``![подпись](attachment:имя.gif)``; этот скрипт находит
такие ссылки, ищет файл среди указанных каталогов и кладёт его base64 в ``attachments``
той же ячейки. После этого медиа видно на GitHub, в Colab и офлайн — без запуска ноутбука
и без похода в сеть.

    python tools/embed_media.py 01-intro/seminar/seminar.ipynb --media assets/anim assets
    python tools/embed_media.py 01-intro/seminar/seminar.ipynb --check

``--check`` ничего не меняет: только сообщает, какие ссылки не разрешаются и какие
вложения лежат без дела. PNG по пути ужимаются до 256-цветной палитры (на схемах курса
это незаметно и экономит больше половины объёма), GIF кладутся как есть.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

REF = re.compile(r"!\[[^\]]*\]\(attachment:([^)\s]+)\)")
MIME = {".png": "image/png", ".gif": "image/gif", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _encode(path: Path) -> tuple[str, str]:
    """(mime, base64). PNG пережимаем в палитру, остальное — как есть."""
    mime = MIME[path.suffix.lower()]
    if path.suffix.lower() == ".png":
        from PIL import Image

        im = Image.open(path).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        buf = io.BytesIO()
        bg.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256).save(buf, "PNG", optimize=True)
        data = buf.getvalue()
    else:
        data = path.read_bytes()
    return mime, base64.b64encode(data).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser(description="Встроить медиа в ноутбук вложениями")
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--media", type=Path, nargs="*", default=[Path("assets/anim"), Path("assets")],
                    help="каталоги, где искать файлы (по умолчанию assets/anim и assets)")
    ap.add_argument("--check", action="store_true", help="ничего не менять, только проверить")
    args = ap.parse_args()

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    missing, embedded, unused, total = [], 0, [], 0

    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        names = set(REF.findall("".join(cell["source"])))
        att = cell.get("attachments", {})
        for name in sorted(set(att) - names):
            unused.append(name)
            if not args.check:
                att.pop(name)
        for name in sorted(names):
            total += 1
            src = next((d / name for d in args.media if (d / name).is_file()), None)
            if src is None:
                missing.append(name)
                continue
            if args.check:
                continue
            mime, payload = _encode(src)
            att[name] = {mime: payload}
            embedded += 1
        if att:
            cell["attachments"] = att
        else:
            cell.pop("attachments", None)

    for name in missing:
        print(f"НЕ НАЙДЕНО: {name}", file=sys.stderr)
    for name in unused:
        print(f"{'лишнее вложение' if args.check else 'убрано лишнее вложение'}: {name}")

    if args.check:
        print(f"ссылок на медиа: {total}, не разрешается: {len(missing)}, лишних вложений: {len(unused)}")
        return 1 if missing or unused else 0
    if missing:
        print("ничего не записано: сначала создайте недостающие файлы", file=sys.stderr)
        return 1

    args.notebook.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    size = args.notebook.stat().st_size / 1024 / 1024
    print(f"встроено медиа: {embedded}; {args.notebook} теперь {size:.2f} МБ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
