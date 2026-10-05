"""Чем платит компьютер за ответ модели: процессор и память.

Два снимка — перед вопросом и после ответа. Загрузку всего компьютера между ними
считает psutil; долю модели — по времени процессора, которое потратили процессы
Ollama: сервер и запущенный им llama-server, который держит модель в памяти.
"""
import platform
import time

import psutil

THREADS = psutil.cpu_count() or 1
CORES = psutil.cpu_count(logical=False) or THREADS


def _cpu_name() -> str:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    except (ImportError, OSError):
        name = platform.processor()
    return name.split(" w/")[0].strip()      # «… w/ Radeon 780M Graphics» — без встроенной графики


CPU_NAME = _cpu_name()
_server: psutil.Process | None = None


def _family() -> list[psutil.Process]:
    """Сервер Ollama и всё, что он запустил. Сервер ищется один раз: перебор всех процессов долгий."""
    global _server
    if _server is None or not _server.is_running():
        _server = None
        for p in psutil.process_iter(["name"]):
            if (p.info["name"] or "").lower() in ("ollama", "ollama.exe"):
                try:
                    if "serve" in p.cmdline():   # «ollama run» в терминале — тоже ollama, но не сервер
                        _server = p
                        break
                except psutil.Error:
                    pass
    if _server is None:
        return []
    try:
        return [_server, *_server.children(recursive=True)]
    except psutil.Error:
        return []


def _busy(procs: list[psutil.Process]) -> dict[int, float]:
    """Сколько секунд процессора уже потратил каждый процесс."""
    busy = {}
    for p in procs:
        try:
            times = p.cpu_times()
            busy[p.pid] = times.user + times.system
        except psutil.Error:
            pass
    return busy


class Usage:
    """Замер на время одного ответа: создать перед вопросом, result() — после ответа."""

    def __init__(self):
        self.before = _busy(_family())
        psutil.cpu_percent(None)                # отсчёт загрузки всего компьютера — с этого момента
        self.started = time.monotonic()

    def result(self) -> dict:
        seconds = time.monotonic() - self.started
        family = _family()
        spent = sum(t - self.before.get(pid, 0) for pid, t in _busy(family).items())
        ram = 0
        for p in family:
            try:
                ram += p.memory_info().rss
            except psutil.Error:
                pass
        memory = psutil.virtual_memory()
        return {
            "cpu": psutil.cpu_percent(None),
            "model_cpu": 100 * spent / seconds / THREADS if seconds else 0,
            "model_ram": ram,
            "ram_used": memory.used, "ram_total": memory.total,
            "cpu_name": CPU_NAME, "cores": CORES, "threads": THREADS,
        }
