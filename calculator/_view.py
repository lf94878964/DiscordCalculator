# -*- coding: utf-8 -*-
"""
_view.py — /計算機 指令的 Components V2 結果顯示介面

版面配置採用與成就系統（AchievementLayout）相同的風格：
    - ui.LayoutView + ui.Container 包裝整個訊息
    - 用按鈕切換「結果」／「詳細步驟」兩個頁籤
    - 「詳細步驟」頁籤內每頁顯示 3 個步驟，並附上該步驟的 LaTeX 圖片
    - 換頁沿用「⏪第一頁 / 🔙上一頁 / 下一頁🔜 / 最後頁⏩」的按鈕列

輸入 / 輸出文字一律用 Discord 的三個反引號（```）包起來顯示成程式碼區塊，
避免長算式跑版，也方便使用者直接複製。
"""

from __future__ import annotations

import io
from typing import List, Optional, Tuple

import discord
from discord import ui


class CalculatorResultView(ui.LayoutView):

    _TAB_RESULT = "result"
    _TAB_STEPS = "steps"

    STEPS_PER_PAGE = 3

    _MODE_LABEL = {"scalar": "純量", "vector": "向量", "matrix": "矩陣"}
    _ANGLE_LABEL = {"degree": "角度", "radian": "弧度"}

    def __init__(
        self,
        bot=None,
        *,
        requester_id: int,
        mode: str,
        angle_unit: str,
        raw_input: str,
        result_text: str,
        result_image: bytes,
        steps: List[Tuple[str, bytes]],
        decimal_text: Optional[str] = None,
        decimal_image: Optional[bytes] = None,
    ):
        super().__init__(timeout=300)
        self.bot = bot
        self.requester_id = requester_id
        self.mode = mode
        self.angle_unit = angle_unit
        self.raw_input = raw_input
        self.result_text = result_text
        self.result_image = result_image
        self.steps = steps  # List[(title, png_bytes)]

        self.decimal_text = decimal_text
        self.decimal_image = decimal_image
        self.show_decimal = False

        self.current_tab = self._TAB_RESULT
        self.step_page = 0

        self._render_seq = 0

        self._build()


    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "這是別人使用 /計算機 的結果，若要查看請自行執行指令喔！",
                ephemeral=True,
            )
            return False
        return True


    @property
    def _max_step_page(self) -> int:
        total = len(self.steps)
        if total <= 0:
            return 0
        return max(0, (total - 1) // self.STEPS_PER_PAGE)

    def _current_step_slice(self) -> List[Tuple[str, bytes]]:
        start = self.step_page * self.STEPS_PER_PAGE
        return self.steps[start: start + self.STEPS_PER_PAGE]


    def _result_filename(self) -> str:
        return f"result_{self._render_seq}.png"

    def _step_filename(self, idx: int) -> str:
        return f"step_{self._render_seq}_{idx}.png"


    def _current_files(self) -> List[discord.File]:
        files: List[discord.File] = []
        if self.current_tab == self._TAB_RESULT:
            active_image = (
                self.decimal_image
                if (self.show_decimal and self.decimal_image is not None)
                else self.result_image
            )
            files.append(discord.File(io.BytesIO(active_image), filename=self._result_filename()))
        else:
            for idx, (_title, img) in enumerate(self._current_step_slice()):
                files.append(discord.File(io.BytesIO(img), filename=self._step_filename(idx)))
        return files



    def _build(self):

        self._render_seq += 1

        self.clear_items()
        container = ui.Container(accent_colour=0x2f9e6b)

        mode_label = self._MODE_LABEL.get(self.mode, self.mode)
        angle_label = self._ANGLE_LABEL.get(self.angle_unit, self.angle_unit)
        has_decimal = self.decimal_image is not None
        showing_decimal = has_decimal and self.show_decimal
        output_text = self.decimal_text if showing_decimal else self.result_text
        output_label = "輸出（小數，最多顯示到小數第12位，超過以...表示）" if showing_decimal else "輸出"
        header = ui.TextDisplay(
            content=(
                f"## <:mura_hide:1429257159618723981>計算機結果\n"
                f"**模式**：{mode_label}　**角度單位**：{angle_label}\n"
                f"**輸入**\n```\n{self.raw_input}\n```\n"
                f"**{output_label}**\n```\n{output_text}\n```\n"
                f"-# 目前發現部分算式的LaTeX渲染有億點點問題，請見諒，如有遇到問題請使用`/report`回報。"
            )
        )
        container.add_item(header)


        tab_row = ui.ActionRow()

        def _style(tab):
            return (discord.ButtonStyle.blurple
                    if self.current_tab == tab
                    else discord.ButtonStyle.gray)

        btn_result = ui.Button(
            label="計算結果",
            style=_style(self._TAB_RESULT),
            disabled=(self.current_tab == self._TAB_RESULT),
        )
        btn_result.callback = self._switch_result

        btn_steps = ui.Button(
            label="詳細步驟",
            style=_style(self._TAB_STEPS),
            disabled=(self.current_tab == self._TAB_STEPS or not self.steps),
        )
        btn_steps.callback = self._switch_steps

        tab_row.add_item(btn_result)
        tab_row.add_item(btn_steps)


        if self.current_tab == self._TAB_RESULT and self.decimal_image is not None:
            btn_decimal = ui.Button(
                label="顯示精確值" if self.show_decimal else "顯示小數",
                style=discord.ButtonStyle.blurple if self.show_decimal else discord.ButtonStyle.gray,
            )
            btn_decimal.callback = self._toggle_decimal
            tab_row.add_item(btn_decimal)

        container.add_item(tab_row)

        container.add_item(ui.Separator())

        if self.current_tab == self._TAB_RESULT:
            container.add_item(
                ui.MediaGallery(discord.MediaGalleryItem(media=f"attachment://{self._result_filename()}"))
            )
        else:
            page_steps = self._current_step_slice()
            if not page_steps:
                container.add_item(ui.TextDisplay(content="（沒有可顯示的步驟）"))
            else:
                for idx, (title, _img) in enumerate(page_steps):
                    step_no = self.step_page * self.STEPS_PER_PAGE + idx + 1
                    container.add_item(
                        ui.TextDisplay(content=f"**步驟 {step_no}：{title}**")
                    )
                    container.add_item(
                        ui.MediaGallery(
                            discord.MediaGalleryItem(media=f"attachment://{self._step_filename(idx)}")
                        )
                    )
                    if idx != len(page_steps) - 1:
                        container.add_item(ui.Separator())

            container.add_item(ui.Separator())
            container.add_item(
                ui.TextDisplay(
                    content=f"第 {self.step_page + 1} / {self._max_step_page + 1} 頁"
                )
            )

            page_row = ui.ActionRow()
            page = self.step_page
            max_p = self._max_step_page

            btn_first = ui.Button(label="⏪第一頁", style=discord.ButtonStyle.gray,
                                   disabled=(page <= 0))
            btn_first.callback = self._goto_first

            btn_prev = ui.Button(label="🔙上一頁", style=discord.ButtonStyle.gray,
                                  disabled=(page <= 0))
            btn_prev.callback = self._goto_prev

            btn_next = ui.Button(label="下一頁🔜", style=discord.ButtonStyle.gray,
                                  disabled=(page >= max_p))
            btn_next.callback = self._goto_next

            btn_last = ui.Button(label="最後頁⏩", style=discord.ButtonStyle.gray,
                                  disabled=(page >= max_p))
            btn_last.callback = self._goto_last

            page_row.add_item(btn_first)
            page_row.add_item(btn_prev)
            page_row.add_item(btn_next)
            page_row.add_item(btn_last)
            container.add_item(page_row)

        self.add_item(container)



    async def _switch_result(self, interaction: discord.Interaction):
        self.current_tab = self._TAB_RESULT
        self._build()
        await interaction.response.edit_message(view=self, attachments=self._current_files())

    async def _switch_steps(self, interaction: discord.Interaction):
        self.current_tab = self._TAB_STEPS
        self.step_page = 0
        self._build()
        await interaction.response.edit_message(view=self, attachments=self._current_files())

    async def _toggle_decimal(self, interaction: discord.Interaction):
        self.show_decimal = not self.show_decimal
        self._build()
        await interaction.response.edit_message(view=self, attachments=self._current_files())



    async def _goto_first(self, interaction: discord.Interaction):
        self.step_page = 0
        self._build()
        await interaction.response.edit_message(view=self, attachments=self._current_files())

    async def _goto_prev(self, interaction: discord.Interaction):
        self.step_page = max(0, self.step_page - 1)
        self._build()
        await interaction.response.edit_message(view=self, attachments=self._current_files())

    async def _goto_next(self, interaction: discord.Interaction):
        self.step_page = min(self._max_step_page, self.step_page + 1)
        self._build()
        await interaction.response.edit_message(view=self, attachments=self._current_files())

    async def _goto_last(self, interaction: discord.Interaction):
        self.step_page = self._max_step_page
        self._build()
        await interaction.response.edit_message(view=self, attachments=self._current_files())