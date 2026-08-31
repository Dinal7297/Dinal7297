"""Cutting List tool definition.

The existing cutting-list behavior remains in the main AI prompt/router.
This module only exposes the tool identity and activation text so the tool is
opt-in and never activated during ordinary FAST/AGENT chat.
"""
CUTTING_TASK_HINT = (
    "TUGAS CUTTING LIST. Prioritaskan ukuran potongan, panjang batang standar, "
    "jumlah batang, packing, efisiensi material, true waste, reusable offcut, "
    "dan anti double-counting. Jangan mengarang ukuran."
)
CUTTING_MENU_TEXT = (
    "✂️ CUTTING LIST AKTIF\n\n"
    "Kirim daftar ukuran/potongan dan panjang batang.\n"
    "Untuk kembali ke AI gunakan /fast atau /agent."
)
