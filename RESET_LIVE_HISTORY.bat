@echo off
cd /d "%~dp0"
if exist "data\live\history.jsonl" del /q "data\live\history.jsonl"
if exist "data\live\latest.json" del /q "data\live\latest.json"
echo ChronoFlux live history cleared.
pause
