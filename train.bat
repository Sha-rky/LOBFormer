@echo off
setlocal enabledelayedexpansion

REM === 設定變數 ===
set base_command=python main.py

REM === 調整的參數組合 ===
set batch_sizes=64 32 128
set lrs=0.0001 0.001
set d_models=128 64 256
set dropouts=0 0.1
set nheads=8 16 32
set num_layers_list=4 2 6 8

REM === 建立資料夾儲存結果 ===
mkdir logs

REM === Grid Search ===
for %%B in (%batch_sizes%) do (
    for %%L in (%lrs%) do (
        for %%D in (%d_models%) do (
            for %%P in (%dropouts%) do (
                for %%H in (%nheads%) do (
                    for %%N in (%num_layers_list%) do (
                        set run_name=bs%%B_lr%%L_dm%%D_dp%%P_nh%%H_nl%%N
                        echo Running %%run_name%%
                        %base_command% --batch_size %%B --learning_rate %%L --d_model %%D --dropout %%P --nhead %%H --num_layers %%N > logs/%%run_name%%.log
                    )
                )
            )
        )
    )
)

echo All experiments done!
pause
