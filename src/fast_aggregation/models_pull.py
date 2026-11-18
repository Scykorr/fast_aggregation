"""Скачивание GGUF с Hugging Face. Сеть нужна только этому модулю.

Ссылка может указывать на файл (blob или resolve) или на репозиторий.
Репозиторий разбирается через API Hugging Face, пользователь выбирает
квантование. Части вида name-00001-of-00003.gguf качаются вместе.
Недокачанный файл лежит рядом как .partial и продолжается заголовком Range.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import httpx

from fast_aggregation.errors import FastAggError

HF_HOSTS = {"huggingface.co", "hf.co"}
SPLIT_NAME = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)
_SHA = re.compile(r"^[0-9a-fA-F]{64}$")
OFFLINE_MESSAGE = (
    "Нет интернета: huggingface.co недоступен. "
    "Скачивание модели — единственное, чему нужна сеть. Остальной кластер работает офлайн."
)
CHUNK = 256 * 1024


@dataclass(frozen=True)
class RemoteFile:
    path: str
    url: str
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class ModelChoice:
    label: str
    size: int | None
    files: tuple[RemoteFile, ...]


@dataclass
class PullSnapshot:
    status: str = "idle"
    name: str = ""
    bytes_done: int = 0
    bytes_total: int | None = None
    speed_bps: float = 0.0
    eta_sec: float | None = None
    error: str | None = None
    paths: list[str] = field(default_factory=list)
    file_index: int = 0
    file_count: int = 0

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "name": self.name,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "speed_bps": round(self.speed_bps, 1),
            "eta_sec": None if self.eta_sec is None else round(self.eta_sec, 1),
            "error": self.error,
            "paths": list(self.paths),
            "file_index": self.file_index,
            "file_count": self.file_count,
        }


class TransferControl:
    def __init__(self) -> None:
        self.running = threading.Event()
        self.running.set()
        self.cancelled = threading.Event()

    def checkpoint(self) -> bool:
        while not self.running.is_set():
            if self.cancelled.is_set():
                return False
            self.running.wait(0.2)
        return not self.cancelled.is_set()

    def cancel(self) -> None:
        self.cancelled.set()

    def pause(self) -> None:
        self.running.clear()

    def resume(self) -> None:
        self.running.set()


@dataclass(frozen=True)
class ParsedLink:
    repo: str
    revision: str
    filename: str | None


def parse_hf_url(text: str) -> ParsedLink:
    raw = (text or "").strip()
    if not raw:
        raise FastAggError("Вставьте ссылку на модель или репозиторий Hugging Face.")
    if "://" not in raw:
        raw = "https://huggingface.co/" + raw.lstrip("/")
    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower().split(":")[0]
    if host not in HF_HOSTS:
        raise FastAggError("Нужна ссылка на huggingface.co.")
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise FastAggError("В ссылке нет репозитория вида организация/имя.")
    repo = parts[0] + "/" + parts[1]
    revision = "main"
    filename = None
    if len(parts) >= 3 and parts[2] in {"blob", "resolve", "tree"}:
        if len(parts) >= 4:
            revision = parts[3]
        if parts[2] in {"blob", "resolve"} and len(parts) >= 5:
            filename = "/".join(parts[4:])
    elif len(parts) > 2:
        raise FastAggError("Не разобрал ссылку Hugging Face. Нужен файл .gguf или страница репозитория.")
    if filename and not filename.lower().endswith(".gguf"):
        raise FastAggError("Файл в ссылке должен быть .gguf.")
    return ParsedLink(repo, revision, filename)


def resolve_url(repo: str, revision: str, path: str) -> str:
    safe_path = quote(path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{quote(revision, safe='')}/{safe_path}"


def tree_api_url(repo: str, revision: str) -> str:
    return f"https://huggingface.co/api/models/{repo}/tree/{quote(revision, safe='')}?recursive=true"


def safe_relative(path: str) -> Path:
    parts = Path(path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise FastAggError(f"Небезопасное имя файла: {path}")
    name = parts[-1]
    if not name.lower().endswith(".gguf"):
        raise FastAggError(f"Ожидался файл .gguf, получено {name}")
    return Path(*parts)


def split_parts(path: str) -> list[str] | None:
    name = Path(path).name
    match = SPLIT_NAME.match(name)
    if not match:
        return None
    total = int(match.group("total"))
    stem = match.group("stem")
    parent = str(Path(path).parent)
    prefix = "" if parent == "." else parent + "/"
    return [f"{prefix}{stem}-{index:05d}-of-{total:05d}.gguf" for index in range(1, total + 1)]


def _sha_from_lfs(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().strip('"')
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    if _SHA.fullmatch(text):
        return text.lower()
    return None


def sha_from_headers(headers: httpx.Headers) -> str | None:
    """sha256 только из ответа huggingface.co до редиректа на CDN.

    ETag после редиректа у xet — это X-Xet-Hash, не sha256. Его нельзя
    сравнивать с lfs.oid. Подходит X-Linked-ETag или X-Checksum-Sha256.
    """

    for key in ("x-linked-etag", "x-checksum-sha256"):
        found = _sha_from_lfs(headers.get(key))
        if found:
            return found
    return None


def content_range_start(header: str) -> int | None:
    text = (header or "").strip().lower()
    if not text.startswith("bytes"):
        return None
    span = text.split(None, 1)[-1].split("/", 1)[0]
    start = span.split("-", 1)[0].strip()
    return int(start) if start.isdigit() else None


def incomplete_body(exc: Exception) -> bool:
    text = str(exc).lower()
    return "peer closed connection" in text or "incomplete" in text or isinstance(exc, httpx.RemoteProtocolError)


def files_from_tree(repo: str, revision: str, payload: object) -> list[RemoteFile]:
    if not isinstance(payload, list):
        raise FastAggError("Hugging Face вернул неожиданный список файлов.")
    files: list[RemoteFile] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("rfilename") or "")
        if not path.lower().endswith(".gguf"):
            continue
        lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else {}
        size = lfs.get("size") if isinstance(lfs, dict) else None
        if not isinstance(size, int):
            size = item.get("size") if isinstance(item.get("size"), int) else None
        sha = _sha_from_lfs(lfs.get("oid")) if isinstance(lfs, dict) else None
        files.append(RemoteFile(path, resolve_url(repo, revision, path), size, sha))
    return files


def choices_from_files(files: list[RemoteFile]) -> list[ModelChoice]:
    grouped: dict[tuple[str, str], list[RemoteFile]] = {}
    order: list[tuple[str, str] | str] = []
    singles: list[RemoteFile] = []
    for item in files:
        match = SPLIT_NAME.match(Path(item.path).name)
        if not match:
            singles.append(item)
            order.append(item.path)
            continue
        key = (str(Path(item.path).parent), match.group("stem") + ":" + match.group("total"))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(item)
    choices: list[ModelChoice] = []
    for key in order:
        if isinstance(key, str):
            item = next(file for file in singles if file.path == key)
            choices.append(ModelChoice(Path(item.path).name, item.size, (item,)))
            continue
        parts = sorted(grouped[key], key=lambda file: file.path)
        total = sum(part.size or 0 for part in parts)
        known = all(part.size is not None for part in parts)
        stem = Path(parts[0].path).name
        match = SPLIT_NAME.match(stem)
        label = f"{match.group('stem')} ({len(parts)} частей)" if match else stem
        choices.append(ModelChoice(label, total if known else None, tuple(parts)))
    return choices


def expand_direct_file(parsed: ParsedLink) -> list[RemoteFile]:
    assert parsed.filename is not None
    names = split_parts(parsed.filename) or [parsed.filename]
    return [RemoteFile(name, resolve_url(parsed.repo, parsed.revision, name)) for name in names]


def group_local_gguf(paths: list[Path]) -> list[list[Path]]:
    """Склеивает части одного split-GGUF. Первой идёт часть 00001."""

    groups: list[list[Path]] = []
    index: dict[tuple[str, str, str], list[Path]] = {}
    for path in paths:
        match = SPLIT_NAME.match(path.name)
        if not match:
            groups.append([path])
            continue
        key = (str(path.parent), match.group("stem"), match.group("total"))
        bucket = index.get(key)
        if bucket is None:
            bucket = []
            index[key] = bucket
            groups.append(bucket)
        bucket.append(path)
    for bucket in groups:
        bucket.sort(key=lambda path: path.name)
    return groups


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_disk_space(folder: Path, need_bytes: int | None, *, free_bytes: int | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free if free_bytes is None else free_bytes
    if need_bytes is None:
        if free < 64 * 1024 * 1024:
            raise FastAggError(f"На диске свободно только {_fmt_bytes(free)}. Для модели места мало.")
        return
    margin = 64 * 1024 * 1024
    if free < need_bytes + margin:
        raise FastAggError(
            f"На диске папки {folder} свободно {_fmt_bytes(free)}, а модель занимает около {_fmt_bytes(need_bytes)}. "
            "Выберите другой диск или освободите место."
        )


def load_token(data_dir: Path) -> str:
    path = data_dir / "hf-token"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def save_token(data_dir: Path, token: str) -> None:
    path = data_dir / "hf-token"
    data_dir.mkdir(parents=True, exist_ok=True)
    cleaned = token.strip()
    if not cleaned:
        if path.exists():
            path.unlink()
        return
    path.write_text(cleaned, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _fmt_bytes(value: int) -> str:
    if value >= 1024 ** 3:
        return f"{value / 1024 ** 3:.1f} ГиБ"
    if value >= 1024 ** 2:
        return f"{value / 1024 ** 2:.1f} МиБ"
    return f"{value} Б"


def _auth_headers(token: str) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _offline(exc: Exception) -> FastAggError:
    return FastAggError(OFFLINE_MESSAGE) if _is_offline(exc) else FastAggError(str(exc))


def _is_offline(exc: Exception) -> bool:
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.ReadTimeout))


def _status_error(response: httpx.Response) -> None:
    if response.status_code in {401, 403}:
        raise FastAggError("Модель закрыта. Введите токен Hugging Face и повторите.")
    if response.status_code == 404:
        raise FastAggError(f"Файл не найден на Hugging Face: {response.request.url}")
    if response.status_code >= 400:
        raise FastAggError(f"Hugging Face ответил {response.status_code}.")


def _total_of(response: httpx.Response, already: int) -> int | None:
    linked = response.headers.get("x-linked-size")
    if linked and linked.isdigit():
        return int(linked)
    content_range = response.headers.get("content-range") or ""
    if "/" in content_range:
        total = content_range.rsplit("/", 1)[-1]
        if total.isdigit():
            return int(total)
    length = response.headers.get("content-length")
    if length and length.isdigit():
        if response.status_code == 206:
            return already + int(length)
        return int(length)
    return None


class ModelPuller:
    def __init__(self, folder: Path, data_dir: Path, client: httpx.Client | None = None) -> None:
        self.folder = folder
        self.data_dir = data_dir
        self.client = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0))
        self._owns_client = client is None
        self.control = TransferControl()
        self.snapshot = PullSnapshot()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._active = 0.0
        self._tick = 0.0
        self._session_bytes = 0

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def token(self) -> str:
        return load_token(self.data_dir)

    def inspect(self, url: str) -> dict:
        parsed = parse_hf_url(url)
        token = self.token()
        try:
            if parsed.filename:
                files = self._head_files(expand_direct_file(parsed), token)
            else:
                response = self.client.get(tree_api_url(parsed.repo, parsed.revision), headers=_auth_headers(token))
                _status_error(response)
                files = files_from_tree(parsed.repo, parsed.revision, response.json())
        except FastAggError:
            raise
        except Exception as exc:
            raise _offline(exc) from exc
        if not files:
            raise FastAggError("В репозитории нет файлов .gguf.")
        choices = choices_from_files(files)
        return {
            "repo": parsed.repo,
            "revision": parsed.revision,
            "choices": [
                {
                    "label": choice.label,
                    "size": choice.size,
                    "files": [
                        {"path": item.path, "url": item.url, "size": item.size, "sha256": item.sha256}
                        for item in choice.files
                    ],
                }
                for choice in choices
            ],
        }

    def start(self, files: list[RemoteFile]) -> dict:
        with self._lock:
            if self.snapshot.status == "running":
                raise FastAggError("Скачивание уже идёт. Поставьте его на паузу или отмените.")
            self.control = TransferControl()
            self.snapshot = PullSnapshot(
                status="running",
                name=Path(files[0].path).name if files else "",
                file_count=len(files),
            )
            self._started = time.monotonic()
            self._active = 0.0
            self._tick = self._started
            self._session_bytes = 0
            self._thread = threading.Thread(target=self._run, args=(list(files),), name="fa-pull", daemon=True)
            self._thread.start()
        return self.view()

    def pause(self) -> dict:
        self.control.pause()
        with self._lock:
            if self.snapshot.status == "running":
                self.snapshot.status = "paused"
        return self.view()

    def resume(self) -> dict:
        self.control.resume()
        with self._lock:
            if self.snapshot.status == "paused":
                self.snapshot.status = "running"
                self._tick = time.monotonic()
        return self.view()

    def cancel(self) -> dict:
        self.control.cancel()
        self.control.resume()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
        with self._lock:
            if self.snapshot.status in {"running", "paused"}:
                self.snapshot.status = "cancelled"
        return self.view()

    def view(self) -> dict:
        with self._lock:
            payload = self.snapshot.to_dict()
        payload["directory"] = str(self.folder)
        return payload

    def _head_files(self, files: list[RemoteFile], token: str) -> list[RemoteFile]:
        headed: list[RemoteFile] = []
        for item in files:
            try:
                response = self.client.head(item.url, headers=_auth_headers(token), follow_redirects=False)
            except Exception as exc:
                raise _offline(exc) from exc
            _status_error(response)
            size = item.size
            linked = response.headers.get("x-linked-size") or response.headers.get("content-length")
            if linked and str(linked).isdigit():
                size = int(linked)
            sha = item.sha256 or sha_from_headers(response.headers)
            headed.append(RemoteFile(item.path, item.url, size, sha))
        return headed

    def _run(self, files: list[RemoteFile]) -> None:
        try:
            known = [item.size for item in files if item.size is not None]
            need = sum(known) if len(known) == len(files) else None
            ensure_disk_space(self.folder, need)
            done_paths: list[str] = []
            carried = 0
            for index, item in enumerate(files, start=1):
                with self._lock:
                    self.snapshot.file_index = index
                    self.snapshot.name = Path(item.path).name
                path = self._download_one(item, carried)
                carried += path.stat().st_size
                done_paths.append(str(path))
            with self._lock:
                self.snapshot.status = "done"
                self.snapshot.paths = done_paths
                self.snapshot.bytes_done = self.snapshot.bytes_total or self.snapshot.bytes_done
                self.snapshot.eta_sec = 0
        except _Cancelled:
            with self._lock:
                self.snapshot.status = "cancelled"
        except FastAggError as exc:
            with self._lock:
                self.snapshot.status = "error"
                self.snapshot.error = str(exc)
        except Exception as exc:
            with self._lock:
                self.snapshot.status = "error"
                self.snapshot.error = str(_offline(exc))

    def _park_bad(self, partial: Path) -> None:
        if not partial.is_file():
            return
        bad = partial.with_name(partial.name[:-8] + ".bad") if partial.name.endswith(".partial") else partial.with_suffix(partial.suffix + ".bad")
        bad.unlink(missing_ok=True)
        partial.replace(bad)

    def _download_one(self, item: RemoteFile, carried: int) -> Path:
        relative = safe_relative(item.path)
        dest = self.folder / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".partial")
        token = self.token()
        attempts = 0
        while True:
            if not self.control.checkpoint():
                raise _Cancelled()
            have = partial.stat().st_size if partial.is_file() else 0
            if dest.is_file() and not partial.is_file():
                if item.sha256 and sha256_file(dest) != item.sha256:
                    dest.unlink()
                else:
                    self._report(carried + dest.stat().st_size, None)
                    return dest
            headers = _auth_headers(token)
            if have:
                headers["Range"] = f"bytes={have}-"
            try:
                with self.client.stream("GET", item.url, headers=headers, follow_redirects=True) as response:
                    if response.status_code == 416 and partial.is_file():
                        self._park_bad(partial)
                        continue
                    _status_error(response)
                    if response.status_code not in {200, 206}:
                        raise FastAggError(f"Hugging Face ответил {response.status_code}.")
                    if response.status_code == 206:
                        start = content_range_start(response.headers.get("content-range") or "")
                        if start is None or start != have:
                            self._park_bad(partial)
                            raise FastAggError(
                                f"Сервер прислал Content-Range с позиции {start}, а на диске уже {have} байт. "
                                "Кусок отложен в .bad, загрузка начнётся заново."
                            )
                    if response.status_code == 200 and have:
                        partial.unlink(missing_ok=True)
                        have = 0
                    total = _total_of(response, have) or item.size
                    mode = "ab" if response.status_code == 206 and have else "wb"
                    if mode == "wb":
                        have = 0
                    paused = False
                    with partial.open(mode) as handle:
                        for chunk in response.iter_bytes(CHUNK):
                            if self.control.cancelled.is_set():
                                raise _Cancelled()
                            if not self.control.running.is_set():
                                paused = True
                                break
                            handle.write(chunk)
                            have += len(chunk)
                            self._session_bytes += len(chunk)
                            overall = carried + have
                            grand = (carried + total) if total is not None else None
                            self._report(overall, grand)
                    if paused:
                        with self._lock:
                            self.snapshot.status = "paused"
                        if not self.control.checkpoint():
                            raise _Cancelled()
                        with self._lock:
                            self.snapshot.status = "running"
                        continue
            except _Cancelled:
                raise
            except FastAggError:
                raise
            except Exception as exc:
                attempts += 1
                if incomplete_body(exc) and attempts < 5:
                    time.sleep(min(8.0, 0.4 * (2 ** (attempts - 1))))
                    continue
                raise _offline(exc) from exc
            if item.sha256:
                digest = sha256_file(partial)
                if digest != item.sha256:
                    self._park_bad(partial)
                    raise FastAggError(
                        "Контрольная сумма sha256 не совпала с oid Hugging Face. "
                        f"Получено {digest}, ожидалось {item.sha256}. Файл отложен как .bad."
                    )
            partial.replace(dest)
            return dest

    def _report(self, done: int, total: int | None) -> None:
        now = time.monotonic()
        with self._lock:
            if self.control.running.is_set():
                self._active += max(0.0, now - self._tick)
            self._tick = now
            self.snapshot.bytes_done = done
            if total is not None:
                self.snapshot.bytes_total = total
            speed = self._session_bytes / self._active if self._active > 0.3 else 0.0
            self.snapshot.speed_bps = speed
            if total is not None and speed > 0 and done <= total:
                self.snapshot.eta_sec = (total - done) / speed
            else:
                self.snapshot.eta_sec = None


class _Cancelled(Exception):
    pass


def files_from_payload(payload: list[dict]) -> list[RemoteFile]:
    files: list[RemoteFile] = []
    for item in payload:
        path = str(item.get("path") or "")
        url = str(item.get("url") or "")
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower().split(":")[0]
        if host not in HF_HOSTS:
            raise FastAggError("Скачивание разрешено только с huggingface.co.")
        size = item.get("size")
        sha = item.get("sha256")
        files.append(
            RemoteFile(
                path,
                url,
                int(size) if isinstance(size, int) else None,
                sha if isinstance(sha, str) and _SHA.fullmatch(sha) else None,
            )
        )
    if not files:
        raise FastAggError("Нечего скачивать.")
    return files
