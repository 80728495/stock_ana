@echo off
if exist "%USERPROFILE%\.ssh\id_ed25519" del "%USERPROFILE%\.ssh\id_ed25519"
if exist "%USERPROFILE%\.ssh\id_ed25519.pub" del "%USERPROFILE%\.ssh\id_ed25519.pub"
ssh-keygen -t ed25519 -C "80728495" -f "%USERPROFILE%\.ssh\id_ed25519" -q -N ""
echo === SSH KEY GENERATED ===
type "%USERPROFILE%\.ssh\id_ed25519.pub"
