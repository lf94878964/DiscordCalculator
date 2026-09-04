# -*- coding: utf-8 -*-
"""
bot.py — 獨立執行版本的進入點。

只做三件事：
    1. 讀取環境變數（Token、選填的測試伺服器 ID）
    2. 建立一個最小化的 discord.py Bot，並載入 `calculator` 這個 cog
    3. 同步斜線指令並開始運行

執行方式：
    1. 安裝依賴：pip install -r requirements.txt
    2. 複製 .env.example 為 .env，填入你的 Bot Token
    3. python bot.py
"""

from __future__ import annotations

import logging
import os

import discord
from discord.ext import commands

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv 是選用的：沒安裝的話，直接讀系統環境變數即可
    pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("calculator-bot")

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
# 選填：若有設定，指令只會同步到這個伺服器（幾乎是即時生效，適合開發測試）。
# 不設定的話會同步成全域指令，最多可能要等一小時才會在所有伺服器出現。
GUILD_ID = os.getenv("GUILD_ID")

# 這個計算機指令用不到任何特殊 Intents（不讀訊息內容、不追蹤成員列表等），
# 有開跟沒開都沒差
intents = discord.Intents.default()


class CalculatorBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self) -> None:
        await self.load_extension("calculator.command")
        logger.info("已載入 calculator cog")

        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            logger.info(f"已同步 {len(synced)} 個指令到測試伺服器（ID: {GUILD_ID}）")
        else:
            synced = await self.tree.sync()
            logger.info(
                f"已同步 {len(synced)} 個全域指令（全域同步可能需要等待最多 1 小時才會在所有伺服器生效）"
            )

    async def on_ready(self) -> None:
        logger.info(f"已登入：{self.user}（ID: {self.user.id}）")


def main() -> None:
    if not TOKEN:
        raise SystemExit(
            "找不到 DISCORD_BOT_TOKEN，請在環境變數或 .env 檔案中設定後再執行。"
            "可參考 .env.example。"
        )

    bot = CalculatorBot()

    try:
        bot.run(TOKEN)
    except discord.LoginFailure:
        raise SystemExit("登入失敗，請確認 DISCORD_BOT_TOKEN 是否正確。")


if __name__ == "__main__":
    main()