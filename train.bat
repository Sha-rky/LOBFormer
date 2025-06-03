@echo off
set base_command=python main.py

REM === 建立資料夾儲存結果 ===
mkdir logs

REM === 直接列出參數組合 ===
call :run 64 0.001 64 4 4
call :run 64 0.001 128 8 4
call :run 64 0.001 128 8 8
call :run 64 0.001 128 8 16
call :run 64 0.0001 64 4 4
call :run 64 0.0001 128 8 4
call :run 64 0.0001 128 8 8
call :run 64 0.0001 128 8 16
REM === 參考基準 ===
call :run 64 0.0001 128 8 4

goto :eof

:run
set bs=%1
set lr=%2
set dm=%3
set nh=%4
set nl=%5
set run_name=bs%bs%_lr%lr%_dm%dm%_nh%nh%_nl%nl%

echo Running bs=%bs% lr=%lr% dm=%dm% nh=%nh% nl=%nl%
%base_command% --batch_size %bs% --learning_rate %lr% --d_model %dm% --nhead %nh% --num_layers %nl% > logs/%run_name%.log
goto :eof
