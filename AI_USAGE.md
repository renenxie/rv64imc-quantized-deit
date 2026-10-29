# AI Usage Log

依課程規定記錄 AI 使用的 prompt 與模型。

| 日期 | 模型 | Prompt（摘要） | 產出 / 用途 |
|---|---|---|---|
| 2026-09-28 | Claude (claude-opus-5-5) | 「幫我建立一個 docker file」 | Dockerfile（RISC-V GCC、Spike、pk 開發環境） |
| 2026-09-28 | Claude (claude-opus-5-5) | 「好那現在建立 baseline kernel」 | bare-metal 執行環境、INT8 FC + requant B0 kernel、測試資料/golden 產生器、instruction mix 腳本、Makefile |
| 2026-09-28 | Claude (claude-opus-5-5) | 「好你先開始進行」（實作剩餘運算） | GELU / 殘差相加 / Softmax / LayerNorm / Attention matmul 的 B0 kernel、Python 整數參考與測試、Makefile 擴充 |
