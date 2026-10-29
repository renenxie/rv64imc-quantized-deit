# data.S — 把模型檔、輸入圖片、參考答案原封不動嵌入執行檔（bare-metal 沒有檔案系統）
# 檔案路徑由組譯參數 -Wa,-I<目錄> 指定搜尋位置

    .section .rodata.model, "a"
    .balign 16
    .global g_model
g_model:
    .incbin "deit_tiny_int8.tflite"

    .section .rodata.io, "a"
    .balign 16
    .global g_input, g_input_end
g_input:
    .incbin "input.bin"
g_input_end:

    .balign 16
    .global g_golden, g_golden_end
g_golden:
    .incbin "golden.bin"
g_golden_end:

# 離線參數表（tools/gen_offline_params 在電腦上產生；TFLM_OFFLINE_PARAMS=2/3 時使用）
    .section .rodata.params, "a"
    .balign 16
    .global g_offline_params, g_offline_params_end
g_offline_params:
    .incbin "offline_params.bin"
g_offline_params_end:
