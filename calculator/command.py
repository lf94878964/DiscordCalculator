# -*- coding: utf-8 -*-
"""
command.py — /calculator（計算機）斜線指令本體。

這是從原本機器人專案中抽出來的獨立版本，移除了原本綁定在那個機器人框架上的
部分（例如封鎖名單檢查、雙語指令註冊 mixin、op 權限檢查、自訂冷卻/計數/日誌
系統），改用 discord.py 內建的機制取代（`app_commands.checks.cooldown`），
讓這個 cog 可以直接掛載到任何一般的 discord.py Bot 上使用。
"""

import discord
import sympy
from discord import app_commands
from discord.ext import commands

from ._engine import (
    CalcError,
    has_decimal_variant,
    safe_calculate,
    to_decimal_display_text,
)
from ._render import render_equation_image, render_expr_image
from ._view import CalculatorResultView


def _format_plain_text(value) -> str:
    try:
        return sympy.pretty(value, use_unicode=True)
    except Exception:
        return str(value)


def _format_decimal_plain_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        widths = [
            max(len(str(row[j])) for row in value)
            for j in range(len(value[0]))
        ] if value else []
        lines = []
        for row in value:
            cells = "  ".join(str(cell).rjust(widths[j]) for j, cell in enumerate(row))
            lines.append(f"[{cells}]")
        return "\n".join(lines)
    return str(value)


class Calculator(commands.Cog):
    """/calculator（計算機）：支援純量／向量／矩陣運算，並可查看詳細計算步驟。"""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="calculator",
        description="數學計算機（支援純量／向量／矩陣，並可查看詳細計算步驟）",
        name_localizations={discord.Locale.taiwan_chinese: "計算機"},
        description_localizations={
            discord.Locale.taiwan_chinese: "數學計算機（支援純量／向量／矩陣，並可查看詳細計算步驟）",
        },
    )
    @app_commands.describe(
        函式="要計算的算式，例如 sin(30)+2^2、[1,2,3]+[4,5,6]、[[1,2],[3,4]].det()",
        角度單位="三角函數的角度單位（預設：弧度）",
        模式="運算模式，若不填會依算式內容自動判斷（有中括號會自動切換成向量／矩陣）",
    )
    @app_commands.choices(
        角度單位=[
            app_commands.Choice(name="弧度", value="radian"),
            app_commands.Choice(name="角度", value="degree"),
        ],
        模式=[
            app_commands.Choice(name="純量", value="scalar"),
            app_commands.Choice(name="向量", value="vector"),
            app_commands.Choice(name="矩陣", value="matrix"),
        ],
    )
    @app_commands.checks.cooldown(1, 3.0, key=lambda i: i.user.id)
    async def calculator(
        self,
        interaction: discord.Interaction,
        函式: str,
        角度單位: app_commands.Choice[str] = None,
        模式: app_commands.Choice[str] = None,
    ):
        raw_input = 函式.strip()
        angle_unit = 角度單位.value if 角度單位 else "radian"
        requested_mode = 模式.value if 模式 else None

        try:
            await interaction.response.defer(ephemeral=False)

            calc_result = await safe_calculate(
                raw_input, requested_mode, angle_unit,
                loop=self.bot.loop,
            )

            original_expr = calc_result.steps[0][2] if calc_result.steps else None

            result_buf = render_equation_image(original_expr, calc_result.result)
            result_bytes = result_buf.read()
            result_text = _format_plain_text(calc_result.result)

            decimal_bytes = None
            decimal_text = None
            if has_decimal_variant(calc_result.result):
                decimal_value = to_decimal_display_text(calc_result.result)
                decimal_img_buf = render_equation_image(original_expr, decimal_value)
                decimal_bytes = decimal_img_buf.read()
                decimal_text = _format_decimal_plain_text(decimal_value)

            step_payload = []
            for title, lhs, rhs in calc_result.steps:
                try:
                    buf = render_equation_image(lhs, rhs)
                    step_payload.append((title, buf.read()))
                except Exception:
                    step_payload.append((title, render_expr_image(str(rhs)).read()))

            view = CalculatorResultView(
                self.bot,
                requester_id=interaction.user.id,
                mode=calc_result.mode,
                angle_unit=angle_unit,
                raw_input=raw_input,
                result_text=result_text,
                result_image=result_bytes,
                steps=step_payload,
                decimal_text=decimal_text,
                decimal_image=decimal_bytes,
            )

            await interaction.followup.send(
                view=view,
                files=view._current_files(),
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except CalcError as e:
            await interaction.followup.send(f"⚠️ {e}", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"抱歉，發生錯誤，稍後再試！ {e}")

    @calculator.error
    async def calculator_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.CommandOnCooldown):
            await interaction.response.send_message(
                f"⏳ 指令冷卻中，請 {error.retry_after:.1f} 秒後再試一次。",
                ephemeral=True,
            )
            return
        raise error


async def setup(bot: commands.Bot):
    await bot.add_cog(Calculator(bot))