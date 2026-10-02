import json
import os
import re
from pathlib import Path

import lmstudio as lms


# =========================
# Konfigurasi
# =========================
IMAGE_PATH = Path("data/raw/nota-sample.png")
OUTPUT_PATH = Path("reports/receipt.json")

MODEL_NAME = os.getenv("LM_STUDIO_MODEL")


# =========================
# Validasi
# =========================
if not MODEL_NAME:
    raise EnvironmentError(
        "Environment variable LM_STUDIO_MODEL belum diset."
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"File gambar tidak ditemukan: {IMAGE_PATH}"
    )


# =========================
# Load image dan model
# =========================
image = lms.prepare_image(str(IMAGE_PATH))
model = lms.llm(MODEL_NAME)


# =========================
# Buat prompt multimodal
# =========================
chat = lms.Chat()

chat.add_user_message(
    """
Baca nota pada gambar dengan teliti.

Ekstrak informasi berikut:
- merchant
- tanggal
- item
- subtotal
- pajak
- total

Keluarkan HANYA JSON valid dengan struktur:

{
  "merchant": "",
  "tanggal": "",
  "item": [
    {
      "nama": "",
      "jumlah": 0,
      "harga": 0
    }
  ],
  "subtotal": 0,
  "pajak": 0,
  "total": 0
}

Aturan:
- Jangan mengarang informasi yang tidak terlihat.
- Jika pajak tidak terlihat, isi 0.
- Jika teks tidak dapat dibaca, gunakan null.
- Nilai uang harus berupa angka, bukan string.
- Jangan gunakan markdown.
- Jangan tambahkan penjelasan di luar JSON.
""",
    images=[image],
)


# =========================
# Jalankan inference
# =========================
prediction = model.respond(chat)

content = prediction.content.strip()


# =========================
# Bersihkan markdown jika ada
# =========================
content = re.sub(
    r"^```(?:json)?\s*|\s*```$",
    "",
    content,
    flags=re.IGNORECASE,
).strip()


# =========================
# Parse JSON
# =========================
try:
    result = json.loads(content)

except json.JSONDecodeError as e:
    print("Output model bukan JSON valid.")
    print("\nOutput asli model:")
    print(content)

    raise ValueError(
        f"Gagal parsing JSON: {e}"
    ) from e


# =========================
# Simpan hasil
# =========================
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# =========================
# Tampilkan hasil
# =========================
print("Hasil ekstraksi nota:")
print(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )
)

print(f"\nDisimpan ke: {OUTPUT_PATH}")
