@echo off
setlocal enabledelayedexpansion

REM === 設定變數 ===
set base_command=python main.py

REM === 調整的參數組合 ===
set batch_sizes=32 64 128
set lrs=0.001 0.0001
set d_models=64 128 256
set dropouts=0.1 0.3
set nheads=4 8
set num_layers_list=2 4 6

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
