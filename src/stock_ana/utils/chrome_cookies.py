"""
Cross-platform Chrome cookie extractor for Gemini authentication.

Reads __Secure-1PSID and __Secure-1PSIDTS directly from Chrome's SQLite
cookie database.

Platform behaviour
------------------
Windows
    Cookie DB may be held with an exclusive lock by Chrome — use Windows
    CreateFile with FILE_SHARE_READ|WRITE|DELETE to bypass it.
    Encryption key is stored in Local State and protected by DPAPI.
    Chrome 127+ App-Bound Encryption (v20) must be disabled via Group Policy
    (ApplicationBoundEncryptionEnabled = 0) before cookies can be read.

macOS
    Chrome does not hold an exclusive lock, so shutil.copy2 works.
    Encryption key is stored in the system Keychain under "Chrome Safe Storage".
    Cookies are AES-128-CBC encrypted (prefix "v10").

Linux
    Chrome does not hold an exclusive lock.
    Encryption key is either a fixed string ("peanuts") or stored in a
    secret-service wallet.  We try both.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

_COOKIE_NAMES = frozenset({"__Secure-1PSID", "__Secure-1PSIDTS"})

# ── Platform detection ────────────────────────────────────────────────────────

_IS_WINDOWS = sys.platform == "win32"
_IS_MAC     = sys.platform == "darwin"
_IS_LINUX   = sys.platform.startswith("linux")


# ══════════════════════════════════════════════════════════════════════════════
#  Windows implementation
# ══════════════════════════════════════════════════════════════════════════════

if _IS_WINDOWS:
    import base64
    import ctypes
    import ctypes.wintypes
    import json as _json

    _kernel32 = ctypes.windll.kernel32
    _GENERIC_READ          = 0x80000000
    _FILE_SHARE_READ       = 0x1
    _FILE_SHARE_WRITE      = 0x2
    _FILE_SHARE_DELETE     = 0x4
    _OPEN_EXISTING         = 3
    _FILE_ATTRIBUTE_NORMAL = 0x80

    def _win_copy_locked(src: Path, dst: Path) -> None:
        """Copy a file Chrome holds with an exclusive lock via WinAPI."""
        _kernel32.CreateFileW.restype = ctypes.c_void_p
        handle = _kernel32.CreateFileW(
            str(src),
            _GENERIC_READ,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
            None, _OPEN_EXISTING, _FILE_ATTRIBUTE_NORMAL, None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle is None or handle == invalid:
            raise OSError(ctypes.GetLastError(), f"Cannot open {src.name}", str(src))
        try:
            size_li = ctypes.c_int64(0)
            if not _kernel32.GetFileSizeEx(handle, ctypes.byref(size_li)):
                raise OSError(ctypes.GetLastError(), "GetFileSizeEx failed", str(src))
            buf = ctypes.create_string_buffer(size_li.value)
            n   = ctypes.wintypes.DWORD(0)
            if not _kernel32.ReadFile(handle, buf, size_li.value, ctypes.byref(n), None):
                raise OSError(ctypes.GetLastError(), "ReadFile failed", str(src))
            dst.write_bytes(buf.raw[: n.value])
        finally:
            _kernel32.CloseHandle(handle)

    class _Blob(ctypes.Structure):
        _fields_ = [("cbData", ctypes.wintypes.DWORD),
                    ("pbData", ctypes.POINTER(ctypes.c_char))]

    def _win_dpapi_decrypt(data: bytes) -> bytes:
        buf      = ctypes.create_string_buffer(data)
        blob_in  = _Blob(len(data), buf)
        blob_out = _Blob()
        ok = ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(blob_in), None, None, None, None, 0,
            ctypes.byref(blob_out),
        )
        if not ok:
            raise OSError(f"CryptUnprotectData failed (WinError {ctypes.GetLastError()})")
        result = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)
        return result

    def _win_get_key(user_data: Path) -> bytes:
        with open(user_data / "Local State", encoding="utf-8") as f:
            state = _json.load(f)
        enc_key = base64.b64decode(state["os_crypt"]["encrypted_key"])[5:]
        return _win_dpapi_decrypt(enc_key)

    def _win_decrypt_cookie(encrypted: bytes, key: bytes) -> str:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if encrypted[:3] == b"v10":
            plaintext = AESGCM(key).decrypt(encrypted[3:15], encrypted[15:], None)
            # Chrome prepends binary metadata — find first fully-printable UTF-8 suffix
            for start in range(len(plaintext)):
                try:
                    val = plaintext[start:].decode("utf-8")
                    if val and all(0x20 <= ord(c) <= 0x7E for c in val):
                        return val
                except UnicodeDecodeError:
                    continue
            raise ValueError("Cannot find printable cookie value in decrypted data")
        if encrypted[:3] == b"v20":
            raise ValueError(
                "Chrome App-Bound Encryption (v20) detected. "
                "Disable via HKLM\\SOFTWARE\\Policies\\Google\\Chrome "
                "ApplicationBoundEncryptionEnabled=0, then re-login to gemini.google.com."
            )
        raise ValueError(f"Unknown cookie encryption prefix: {encrypted[:3]!r}")

    def _get_gemini_cookies_windows(profile: str = "Default") -> dict[str, str]:
        import subprocess as _sp
        localappdata = Path(os.environ.get("LOCALAPPDATA", "C:/Users/Public"))
        user_data    = localappdata / "Google" / "Chrome" / "User Data"
        if not user_data.exists():
            raise FileNotFoundError(f"Chrome user data not found: {user_data}")
        key = _win_get_key(user_data)
        db  = next(
            (p for p in [
                user_data / profile / "Network" / "Cookies",
                user_data / profile / "Cookies",
            ] if p.exists()), None,
        )
        if db is None:
            raise FileNotFoundError(f"Chrome Cookies DB not found (profile='{profile}')")

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()

        # 第一次尝试：Chrome 可能没开，直接用共享模式拷
        chrome_was_killed = False
        try:
            _win_copy_locked(db, Path(tmp.name))
        except OSError as e:
            # errno 32 = Chrome 正在运行持有排他锁 → 强制关闭后再读
            if e.errno != 32 and e.args[0] != 32:
                raise
            import time as _time
            _sp.run(
                ["taskkill", "/F", "/IM", "chrome.exe"],
                capture_output=True,
            )
            chrome_was_killed = True
            import logging as _logging
            _logging.getLogger(__name__).info(
                "Chrome 已关闭（Cookie 文件锁），读取完成后不会自动重启"
            )
            _time.sleep(3)  # 等 SQLite WAL checkpoint 完成
            shutil.copy2(db, tmp.name)  # Chrome 已关，直接拷

        # 同时拷 WAL / SHM（Chrome 开着时新 cookie 在 WAL 里）
        for _suffix in ("-wal", "-shm"):
            _src = Path(str(db) + _suffix)
            if _src.exists():
                try:
                    if chrome_was_killed:
                        shutil.copy2(_src, tmp.name + _suffix)
                    else:
                        _win_copy_locked(_src, Path(tmp.name + _suffix))
                except Exception:
                    pass

        result: dict[str, str] = {}
        try:
            con  = sqlite3.connect(tmp.name)
            ph   = ",".join("?" * len(_COOKIE_NAMES))
            rows = con.execute(
                f"SELECT name, encrypted_value FROM cookies"
                f" WHERE host_key LIKE '%google.com%' AND name IN ({ph})",
                list(_COOKIE_NAMES),
            ).fetchall()
            con.close()
            for name, enc in rows:
                if enc:
                    result[name] = _win_decrypt_cookie(bytes(enc), key)
        finally:
            os.unlink(tmp.name)
            for _suffix in ("-wal", "-shm"):
                _p = tmp.name + _suffix
                if os.path.exists(_p):
                    os.unlink(_p)
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  macOS implementation
# ══════════════════════════════════════════════════════════════════════════════

def _mac_get_key() -> bytes:
    """Retrieve Chrome's AES-128 key from the macOS Keychain."""
    import subprocess
    from hashlib import pbkdf2_hmac
    result   = subprocess.run(
        ["security", "find-generic-password", "-w",
         "-s", "Chrome Safe Storage", "-a", "Chrome"],
        capture_output=True, text=True,
    )
    password = result.stdout.strip().encode() if result.returncode == 0 else b"peanuts"
    return pbkdf2_hmac("sha1", password, b"saltysalt", iterations=1003, dklen=16)


def _mac_decrypt_cookie(encrypted: bytes, key: bytes) -> str:
    """Decrypt a macOS Chrome v10 AES-128-CBC cookie."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    if encrypted[:3] == b"v10":
        iv     = b" " * 16
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        dec    = cipher.decryptor()
        padded = dec.update(encrypted[3:]) + dec.finalize()
        return padded[: -padded[-1]].decode("utf-8")  # PKCS7 unpad
    raise ValueError(f"Unknown cookie encryption prefix: {encrypted[:3]!r}")


def _get_gemini_cookies_mac(profile: str = "Default") -> dict[str, str]:
    user_data = Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
    if not user_data.exists():
        raise FileNotFoundError(f"Chrome user data not found: {user_data}")
    key = _mac_get_key()
    db  = next(
        (p for p in [
            user_data / profile / "Network" / "Cookies",
            user_data / profile / "Cookies",
        ] if p.exists()), None,
    )
    if db is None:
        raise FileNotFoundError(f"Chrome Cookies DB not found (profile='{profile}')")
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    shutil.copy2(db, tmp.name)  # macOS does not exclusively lock the file
    result: dict[str, str] = {}
    try:
        con  = sqlite3.connect(tmp.name)
        ph   = ",".join("?" * len(_COOKIE_NAMES))
        rows = con.execute(
            f"SELECT name, encrypted_value FROM cookies"
            f" WHERE host_key LIKE '%google.com%' AND name IN ({ph})",
            list(_COOKIE_NAMES),
        ).fetchall()
        con.close()
        for name, enc in rows:
            if enc:
                result[name] = _mac_decrypt_cookie(bytes(enc), key)
    finally:
        os.unlink(tmp.name)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Linux implementation
# ══════════════════════════════════════════════════════════════════════════════

def _linux_get_key() -> bytes:
    """Try secret-service wallet first, then fall back to fixed 'peanuts' key."""
    from hashlib import pbkdf2_hmac
    password = b"peanuts"
    try:
        import secretstorage
        bus = secretstorage.dbus_init()
        col = secretstorage.get_default_collection(bus)
        for item in col.get_all_items():
            if "Chrome" in (item.get_label() or ""):
                password = item.get_secret()
                break
    except Exception:
        pass
    return pbkdf2_hmac("sha1", password, b"saltysalt", iterations=1, dklen=16)


def _linux_decrypt_cookie(encrypted: bytes, key: bytes) -> str:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    if encrypted[:3] == b"v10":
        iv     = b" " * 16
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        dec    = cipher.decryptor()
        padded = dec.update(encrypted[3:]) + dec.finalize()
        return padded[: -padded[-1]].decode("utf-8")
    raise ValueError(f"Unknown cookie encryption prefix: {encrypted[:3]!r}")


def _get_gemini_cookies_linux(profile: str = "Default") -> dict[str, str]:
    user_data = next(
        (p for p in [
            Path.home() / ".config" / "google-chrome",
            Path.home() / ".config" / "chromium",
        ] if p.exists()),
        Path.home() / ".config" / "google-chrome",
    )
    if not user_data.exists():
        raise FileNotFoundError(f"Chrome user data not found: {user_data}")
    key = _linux_get_key()
    db  = next(
        (p for p in [
            user_data / profile / "Network" / "Cookies",
            user_data / profile / "Cookies",
        ] if p.exists()), None,
    )
    if db is None:
        raise FileNotFoundError(f"Chrome Cookies DB not found (profile='{profile}')")
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    shutil.copy2(db, tmp.name)
    result: dict[str, str] = {}
    try:
        con  = sqlite3.connect(tmp.name)
        ph   = ",".join("?" * len(_COOKIE_NAMES))
        rows = con.execute(
            f"SELECT name, encrypted_value FROM cookies"
            f" WHERE host_key LIKE '%google.com%' AND name IN ({ph})",
            list(_COOKIE_NAMES),
        ).fetchall()
        con.close()
        for name, enc in rows:
            if enc:
                result[name] = _linux_decrypt_cookie(bytes(enc), key)
    finally:
        os.unlink(tmp.name)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Public API — platform dispatch
# ══════════════════════════════════════════════════════════════════════════════

def get_gemini_cookies(profile: str = "Default") -> dict[str, str]:
    """
    Extract Gemini authentication cookies from Chrome.

    Returns a dict containing any of:
        '__Secure-1PSID'
        '__Secure-1PSIDTS'

    Dispatches to the appropriate platform implementation automatically.
    Raises FileNotFoundError if Chrome is not installed.
    """
    if _IS_WINDOWS:
        return _get_gemini_cookies_windows(profile)
    if _IS_MAC:
        return _get_gemini_cookies_mac(profile)
    if _IS_LINUX:
        return _get_gemini_cookies_linux(profile)
    raise NotImplementedError(f"Unsupported platform: {sys.platform}")
