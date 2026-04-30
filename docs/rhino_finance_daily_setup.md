# RhinoFinance 每日视频下载转写自动化 — Windows 部署指南

## 概述

本流程每天自动执行以下任务：
1. **获取最新视频** — 从 YouTube `@RhinoFinance` 频道获取最近 5 个视频，找到第一个未处理的
2. **下载音频** — 使用 `yt-dlp` + cookie 登录态下载 m4a 音频
3. **语音转写** — 使用 `faster-whisper`（Whisper small 模型）将音频转为文字（txt + srt）
4. **AI 总结** — 调用火山引擎大模型 API 对转写内容生成结构化摘要
5. **飞书推送** — 将总结结果以富文本消息推送到飞书
6. **失败告警** — 任何阶段失败都会通过飞书发送告警消息

### 流程图

```
cron 定时触发 (每天 11:00)
      │
      ▼
rhino_finance_daily.sh  ← 入口 shell 脚本，设置环境变量和代理
      │
      ▼
rhino_finance_daily.py  ← 主控 Python 脚本
      │
      ├─ 1. get_today_video()
      │     └─ yt-dlp --flat-playlist 获取频道最新 5 个视频
      │     └─ 和 status.json 比对，找出未处理的视频
      │
      ├─ 2. download_audio()
      │     └─ 调用 yt_audio.py 下载音频 → ~/Music/yt_audio/
      │
      ├─ 3. transcribe_audio()
      │     └─ 调用 yt_audio.py --transcribe → ~/Documents/yt_transcripts/
      │     └─ 底层: faster-whisper (small 模型, 中文)
      │
      ├─ 4. summarize_transcript()
      │     └─ 调用火山引擎 LLM API 生成结构化总结
      │
      └─ 5. send_feishu_message()
            └─ 飞书机器人推送总结结果
```

---

## 文件结构

在 Windows 上建议将所有文件放在同一个工作目录下，例如 `C:\Users\<你的用户名>\rhino_finance\`。

```
rhino_finance\
├── rhino_finance_daily.py      # 主控脚本
├── yt_audio.py                 # 下载 & 转写工具脚本
├── cookies.txt                 # YouTube cookie 文件（运行时自动生成）
├── run_daily.bat               # Windows 定时任务启动脚本（替代 .sh）
```

运行时生成的数据目录：

```
%USERPROFILE%\Music\yt_audio\           # 下载的音频文件
%USERPROFILE%\Documents\yt_transcripts\ # 转写输出（.txt + .srt）
%TEMP%\openclaw\rhino_finance_daily.log # 运行日志
%TEMP%\rhino_finance_status.json        # 已处理视频 ID 记录
```

---

## 环境准备

### 1. 安装 Python 3.10+

从 https://www.python.org/downloads/ 下载安装，**安装时勾选 "Add Python to PATH"**。

验证：
```powershell
python --version
```

### 2. 安装 yt-dlp

```powershell
pip install yt-dlp
```

验证：
```powershell
yt-dlp --version
```

### 3. 安装 faster-whisper

```powershell
pip install faster-whisper
```

> **注意**：faster-whisper 需要 CUDA 支持才能用 GPU 加速。如果没有 NVIDIA 显卡，会自动使用 CPU（速度较慢但可用）。
>
> 如果碰到 `ctranslate2` 相关错误，需另外安装：
> ```powershell
> pip install ctranslate2
> ```

### 4. 安装 ffmpeg

faster-whisper 和 yt-dlp 都需要 ffmpeg。

方式一（推荐）— 使用 winget：
```powershell
winget install ffmpeg
```

方式二 — 手动下载：
从 https://www.gyan.dev/ffmpeg/builds/ 下载，解压后将 `bin` 目录添加到系统 PATH。

验证：
```powershell
ffmpeg -version
```

---

## 脚本部署

### 1. 复制 `rhino_finance_daily.py`

从 Mac 上复制 `/Users/wl/rhino_finance_daily.py` 到 Windows 工作目录。

**需要修改的配置项**（文件开头的常量）：

```python
# ========== 需要修改的路径 ==========
AUDIO_DIR = os.path.expanduser("~/Music/yt_audio")           # 可保持不变，Windows 会自动展开
TRANSCRIPT_DIR = os.path.expanduser("~/Documents/yt_transcripts")  # 同上
YT_SCRIPT = r"C:\Users\<你的用户名>\rhino_finance\yt_audio.py"    # ← 改成实际路径
YT_PYTHON = "python"                                          # ← Windows 上通常直接用 python
```

**yt-dlp 路径修改**（在 `get_today_video()` 函数中）：

```python
# Mac 上是 "/opt/homebrew/bin/yt-dlp"
# Windows 上改为：
cmd = [
    "yt-dlp",          # ← 如果已加入 PATH，直接用命令名
    "--proxy", PROXY,
    ...
]
```

**代理配置**（如果 Windows 上的代理端口不同）：

```python
PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or "http://127.0.0.1:5782"
# ↑ 改成你 Windows 上的代理地址和端口
```

### 2. 复制 `yt_audio.py`

从 Mac 上复制 `/Users/wl/gem_claude/yt_audio.py` 到 Windows 工作目录。

此脚本无需修改，路径均使用 `Path.home()` 自动适配。

### 3. 导出 YouTube Cookies

在 Windows 上先用 Chrome 登录 YouTube，然后运行：

```powershell
cd C:\Users\<你的用户名>\rhino_finance
python yt_audio.py --export-cookies
```

这会在当前目录生成 `cookies.txt`。如果 RhinoFinance 频道有付费内容，需要用已订阅的账号登录。

> **Cookie 有效期有限**，失效后需重新导出。脚本会在下载失败时自动检测 cookie 问题并告警。

### 4. 创建启动脚本 `run_daily.bat`

```bat
@echo off
chcp 65001 >nul
setlocal

:: 代理配置（按需修改）
set HTTPS_PROXY=http://127.0.0.1:5782
set HTTP_PROXY=http://127.0.0.1:5782

:: 日志目录
set LOG_DIR=%TEMP%\openclaw
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set LOG_FILE=%LOG_DIR%\rhino_finance_daily.log

:: 切换到脚本目录
cd /d C:\Users\<你的用户名>\rhino_finance

echo. >> "%LOG_FILE%"
echo ════════════════════════════════════════ >> "%LOG_FILE%"
python rhino_finance_daily.py >> "%LOG_FILE%" 2>&1

endlocal
```

---

## 配置 Windows 定时任务

使用 **任务计划程序 (Task Scheduler)** 替代 macOS 的 cron。

### GUI 方式

1. 按 `Win + R`，输入 `taskschd.msc`，回车
2. 右侧点击 **"创建基本任务"**
3. 名称：`RhinoFinance 每日视频转写`
4. 触发器：**每天**，时间设为 `11:00`（与 Mac 上一致）
5. 操作：**启动程序**
   - 程序或脚本：`C:\Users\<你的用户名>\rhino_finance\run_daily.bat`
   - 起始于：`C:\Users\<你的用户名>\rhino_finance`
6. 勾选 **"不管用户是否登录都要运行"**（可选，需要输入密码）
7. 勾选 **"使用最高权限运行"**

### 命令行方式

```powershell
schtasks /create /tn "RhinoFinance Daily" /tr "C:\Users\<你的用户名>\rhino_finance\run_daily.bat" /sc daily /st 11:00 /rl highest
```

---

## API 密钥配置

脚本中硬编码了以下密钥，部署到 Windows 后建议改为从环境变量读取：

| 用途 | 变量名（脚本中） | 说明 |
|------|-------------------|------|
| 火山引擎 LLM | `LLM_API_KEY` | 用于 AI 总结 |
| 飞书 App | `FEISHU_APP_ID` / `FEISHU_APP_SECRET` | 用于推送消息 |
| 飞书收信人 | `FEISHU_USER_OPEN_ID` | 消息接收者 |

**推荐做法** — 在 Windows 系统环境变量或 `.env` 文件中配置，避免明文存储在代码里：

```python
# 修改脚本中的配置为：
LLM_API_KEY = os.environ.get("RHINO_LLM_API_KEY", "你的默认值")
FEISHU_APP_ID = os.environ.get("RHINO_FEISHU_APP_ID", "你的默认值")
FEISHU_APP_SECRET = os.environ.get("RHINO_FEISHU_APP_SECRET", "你的默认值")
```

---

## 验证与调试

### 手动测试完整流程

```powershell
cd C:\Users\<你的用户名>\rhino_finance
python rhino_finance_daily.py
```

### 分步测试

```powershell
# 1. 测试 yt-dlp 是否能访问频道
yt-dlp --proxy http://127.0.0.1:5782 --flat-playlist --playlist-end 3 --print "%(id)s %(title)s" "https://www.youtube.com/@RhinoFinance/videos"

# 2. 测试下载音频
python yt_audio.py "https://www.youtube.com/watch?v=<视频ID>"

# 3. 测试转写
python yt_audio.py --transcribe --model small "%USERPROFILE%\Music\yt_audio\<音频文件名>.m4a"
```

### 查看日志

```powershell
type %TEMP%\openclaw\rhino_finance_daily.log
```

### 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `yt-dlp` 超时 | 代理未启动或端口错误 | 检查代理配置和网络 |
| Cookie 失效 | YouTube 登录态过期 | 重新运行 `python yt_audio.py --export-cookies` |
| 转写极慢 | 无 GPU，CPU 运行 whisper | 换用 `tiny` 或 `base` 模型，或安装 CUDA |
| `ModuleNotFoundError: faster_whisper` | 未安装依赖 | `pip install faster-whisper` |
| `ffmpeg not found` | ffmpeg 未安装或未加入 PATH | 安装 ffmpeg 并确认 PATH |
| 飞书推送失败 | token 过期或网络问题 | 检查 App ID/Secret 是否正确 |

---

## 与 Mac 的差异总结

| 项目 | Mac (当前) | Windows (部署) |
|------|-----------|---------------|
| 定时任务 | `crontab` (`0 11 * * *`) | 任务计划程序 (`schtasks`) |
| 启动脚本 | `rhino_finance_daily.sh` | `run_daily.bat` |
| Python | `/Users/wl/.pyenv/shims/python3` | `python`（PATH 中） |
| yt-dlp | `/opt/homebrew/bin/yt-dlp` | `yt-dlp`（pip 安装后在 PATH 中） |
| yt_audio.py 位置 | `/Users/wl/gem_claude/yt_audio.py` | 自行指定，建议与主脚本同目录 |
| Cookie 文件 | `~/gem_claude/cookies.txt` | 与 `yt_audio.py` 同目录 |
| 状态文件 | `/tmp/rhino_finance_status.json` | `%TEMP%\rhino_finance_status.json` |
| 日志 | `/tmp/openclaw/` | `%TEMP%\openclaw\` |
