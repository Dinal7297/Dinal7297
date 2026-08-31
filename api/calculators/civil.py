"""Civil Calculator local engine for Designmanufaktur.

This module is deliberately isolated from the normal AI chat path. It is
executed only when /civil is activated.
"""
import math
import re

def parse_number(value):
    """
    Mengubah angka seperti:
    5
    5.5
    5,5
    1.000
    1,000
    menjadi float.
    """

    if value is None:
        return None

    value = str(value).strip()

    value = value.replace(" ", "")

    if "," in value and "." in value:

        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "")
            value = value.replace(",", ".")
        else:
            value = value.replace(",", "")

    elif "," in value:

        value = value.replace(",", ".")

    try:
        return float(value)

    except Exception:
        return None


def fmt(value, decimals=3):

    if value is None:
        return "-"

    if abs(value - round(value)) < 0.000001:
        return str(int(round(value)))

    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


# ============================================================
# DIMENSION PARSER
# ============================================================

def extract_dimensions(text):

    t = text.lower()

    patterns = [
        r"(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)",
        r"(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)",
    ]

    for pattern in patterns:

        m = re.search(pattern, t)

        if m:

            values = [
                parse_number(x)
                for x in m.groups()
            ]

            return values

    return []


def extract_value_with_unit(text, units):

    unit_pattern = "|".join(
        re.escape(x)
        for x in units
    )

    pattern = (
        r"(\d+(?:[.,]\d+)?)"
        r"\s*("
        + unit_pattern
        + r")"
    )

    m = re.search(
        pattern,
        text.lower()
    )

    if not m:
        return None

    value = parse_number(m.group(1))
    unit = m.group(2)

    if value is None:
        return None

    return value, unit


def to_meter(value, unit):

    unit = unit.lower()

    if unit in ("mm",):
        return value / 1000

    if unit in ("cm",):
        return value / 100

    if unit in ("m", "meter", "meters"):
        return value

    return value


# ============================================================
# CIVIL CONCRETE CALCULATOR
# ============================================================

def concrete_materials(
    volume,
    mix=(1, 2, 3),
    dry_factor=1.54,
    cement_density=1440,
    sack_weight=50,
    wc=0.50,
):

    a, b, c = mix

    total_parts = a + b + c

    dry_volume = volume * dry_factor

    cement_volume = (
        dry_volume * a / total_parts
    )

    sand_volume = (
        dry_volume * b / total_parts
    )

    gravel_volume = (
        dry_volume * c / total_parts
    )

    cement_kg = (
        cement_volume * cement_density
    )

    cement_sacks = math.ceil(
        cement_kg / sack_weight
    )

    water_liter = cement_kg * wc

    return {
        "volume": volume,
        "dry_volume": dry_volume,
        "cement_kg": cement_kg,
        "cement_sacks": cement_sacks,
        "sand_m3": sand_volume,
        "gravel_m3": gravel_volume,
        "water_liter": water_liter,
    }


def civil_concrete_calculation(
    text,
    title="KEBUTUHAN BETON"
):

    dims = extract_dimensions(text)

    if len(dims) == 3:

        p, l, t = dims

    else:

        return None

    # Jika ukuran terlihat seperti cm/mm,
    # pengguna sebaiknya menyebut unit.
    # Default hanya meter jika tidak ada unit.

    volume = p * l * t

    result = concrete_materials(
        volume
    )

    answer = f"""
🏗️ {title}

📋 DATA

• Panjang: {fmt(p)} m
• Lebar: {fmt(l)} m
• Tebal/Tinggi: {fmt(t)} m
• Volume beton: {fmt(volume)} m³

⚙️ ASUMSI

• Campuran nominal: 1 : 2 : 3
• Faktor volume kering: 1,54
• Berat semen: 1.440 kg/m³
• 1 zak semen: 50 kg
• Air: w/c sekitar 0,50

🧮 PERHITUNGAN

• Semen: sekitar {fmt(result["cement_kg"])} kg
• Semen: sekitar {result["cement_sacks"]} zak
• Pasir: sekitar {fmt(result["sand_m3"])} m³
• Kerikil: sekitar {fmt(result["gravel_m3"])} m³
• Air: sekitar {fmt(result["water_liter"])} liter

🔍 VALIDASI

• Volume = panjang × lebar × tebal
• {fmt(p)} × {fmt(l)} × {fmt(t)} = {fmt(volume)} m³
• Estimasi material menggunakan asumsi 1 : 2 : 3

📊 RINGKASAN

• Beton: {fmt(volume)} m³
• Semen: {result["cement_sacks"]} zak
• Pasir: {fmt(result["sand_m3"])} m³
• Kerikil: {fmt(result["gravel_m3"])} m³
• Air: {fmt(result["water_liter"])} liter

📝 CATATAN

Hasil ini adalah estimasi kebutuhan material.
Untuk pekerjaan struktur penting, gunakan mix design
dan verifikasi engineer/insinyur struktur.
"""

    return answer.strip()


# ============================================================
# CIVIL WALL CALCULATOR
# ============================================================

def civil_wall_calculation(text):

    dims = extract_dimensions(text)

    if len(dims) < 2:
        return None

    p, l = dims[0], dims[1]

    t = dims[2] if len(dims) >= 3 else None

    area = p * l

    lower = text.lower()

    if "batako" in lower:

        material_name = "batako"

        # Estimasi umum.
        # Nilai dapat berbeda sesuai ukuran batako.
        pieces_per_m2 = 12.5

    else:

        material_name = "bata"

        pieces_per_m2 = 50

    pieces = math.ceil(
        area * pieces_per_m2
    )

    # Mortar sederhana:
    mortar_per_m2 = 0.02

    mortar_volume = area * mortar_per_m2

    # Estimasi campuran mortar 1:4
    dry_factor = 1.33

    dry_mortar = (
        mortar_volume * dry_factor
    )

    cement_volume = dry_mortar / 5

    cement_kg = cement_volume * 1440

    cement_sacks = math.ceil(
        cement_kg / 50
    )

    sand = (
        dry_mortar * 4 / 5
    )

    return f"""
🏗️ KEBUTUHAN DINDING

📋 DATA

• Panjang dinding: {fmt(p)} m
• Tinggi dinding: {fmt(l)} m
• Luas dinding: {fmt(area)} m²
• Material: {material_name}

⚙️ ASUMSI

• Bata: sekitar 50 buah/m²
• Batako: sekitar 12,5 buah/m²
• Mortar: sekitar 0,02 m³/m²
• Campuran mortar: 1 : 4
• Faktor volume kering mortar: 1,33

🧮 PERHITUNGAN

• Luas = {fmt(p)} × {fmt(l)}
• Luas = {fmt(area)} m²
• {material_name}: sekitar {pieces} buah
• Mortar: sekitar {fmt(mortar_volume)} m³
• Semen: sekitar {fmt(cement_kg)} kg
• Semen: sekitar {cement_sacks} zak
• Pasir mortar: sekitar {fmt(sand)} m³

🔍 VALIDASI

• Luas dinding dihitung dari panjang × tinggi
• Jumlah material dibulatkan ke atas
• Angka adalah estimasi dan tergantung ukuran material nyata

📊 RINGKASAN

• Luas: {fmt(area)} m²
• {material_name}: sekitar {pieces} buah
• Semen: sekitar {cement_sacks} zak
• Pasir mortar: sekitar {fmt(sand)} m³

📝 CATATAN

Jumlah aktual dapat berubah karena ukuran bata/batako,
ketebalan nat, bukaan pintu/jendela, pecahan, dan metode
pemasangan.
""".strip()


# ============================================================
# PLASTER CALCULATOR
# ============================================================

def civil_plaster_calculation(text):

    dims = extract_dimensions(text)

    if len(dims) < 2:
        return None

    p, h = dims[0], dims[1]

    area = p * h

    thickness = 0.015

    thickness_data = extract_value_with_unit(
        text,
        ["mm", "cm", "m"]
    )

    if thickness_data:

        value, unit = thickness_data

        candidate = to_meter(
            value,
            unit
        )

        if 0.003 <= candidate <= 0.1:
            thickness = candidate

    volume = area * thickness

    dry_factor = 1.33

    dry_volume = volume * dry_factor

    # 1:4
    cement_volume = dry_volume / 5

    sand_volume = (
        dry_volume * 4 / 5
    )

    cement_kg = (
        cement_volume * 1440
    )

    cement_sacks = math.ceil(
        cement_kg / 50
    )

    return f"""
🏗️ KEBUTUHAN PLESTERAN

📋 DATA

• Panjang: {fmt(p)} m
• Tinggi: {fmt(h)} m
• Luas: {fmt(area)} m²
• Tebal plester: {fmt(thickness * 1000)} mm

⚙️ ASUMSI

• Campuran mortar: 1 : 4
• Faktor volume kering: 1,33
• Berat semen: 1.440 kg/m³
• 1 zak semen: 50 kg

🧮 PERHITUNGAN

• Volume basah: {fmt(volume)} m³
• Volume kering: {fmt(dry_volume)} m³
• Semen: sekitar {fmt(cement_kg)} kg
• Semen: sekitar {cement_sacks} zak
• Pasir: sekitar {fmt(sand_volume)} m³

📊 RINGKASAN

• Luas: {fmt(area)} m²
• Semen: sekitar {cement_sacks} zak
• Pasir: sekitar {fmt(sand_volume)} m³
• Tebal: {fmt(thickness * 1000)} mm

📝 CATATAN

Hasil adalah estimasi. Konsumsi aktual dipengaruhi
ketebalan, permukaan dinding, campuran, dan kehilangan
material di lapangan.
""".strip()


# ============================================================
# ACIAN CALCULATOR
# ============================================================

def civil_acian_calculation(text):

    dims = extract_dimensions(text)

    if len(dims) < 2:
        return None

    p, h = dims[0], dims[1]

    area = p * h

    # Estimasi konsumsi acian:
    # sekitar 1,5 kg/m2/mm
    thickness_mm = 2

    thickness_data = extract_value_with_unit(
        text,
        ["mm", "cm"]
    )

    if thickness_data:

        value, unit = thickness_data

        if unit == "cm":
            value *= 10

        if 1 <= value <= 10:
            thickness_mm = value

    consumption = (
        1.5 * thickness_mm
    )

    powder_kg = (
        area * consumption
    )

    bags = math.ceil(
        powder_kg / 40
    )

    return f"""
🏗️ KEBUTUHAN ACIAN

📋 DATA

• Panjang: {fmt(p)} m
• Tinggi: {fmt(h)} m
• Luas: {fmt(area)} m²
• Tebal acian: sekitar {fmt(thickness_mm)} mm

⚙️ ASUMSI

• Konsumsi acian: sekitar 1,5 kg/m²/mm
• Kemasan estimasi: 40 kg/zak

🧮 PERHITUNGAN

• Kebutuhan: {fmt(powder_kg)} kg
• Perkiraan: {bags} zak

📊 RINGKASAN

• Luas: {fmt(area)} m²
• Acian: sekitar {fmt(powder_kg)} kg
• Estimasi kemasan 40 kg: {bags} zak

📝 CATATAN

Konsumsi aktual mengikuti produk acian yang digunakan,
ketebalan aplikasi, dan kondisi permukaan.
""".strip()


# ============================================================
# EXCAVATION / URUGAN
# ============================================================

def civil_volume_calculation(
    text,
    title,
    material_name
):

    dims = extract_dimensions(text)

    if len(dims) < 3:
        return None

    p, l, t = dims[:3]

    volume = p * l * t

    return f"""
🏗️ {title}

📋 DATA

• Panjang: {fmt(p)} m
• Lebar: {fmt(l)} m
• Kedalaman/Tebal: {fmt(t)} m

🧮 PERHITUNGAN

Volume =
panjang × lebar × kedalaman

= {fmt(p)} × {fmt(l)} × {fmt(t)}

= {fmt(volume)} m³

📊 RINGKASAN

• {material_name}: {fmt(volume)} m³

📝 CATATAN

Angka merupakan volume geometris.
Kebutuhan pembelian aktual dapat berbeda karena
pemadatan, swell, penyusutan, dan kehilangan material.
""".strip()


# ============================================================
# REBAR CALCULATOR
# ============================================================

def civil_rebar_calculation(text):

    t = text.lower()

    diameter_match = re.search(
        r"(?:d|dia|diameter|besi)\s*"
        r"(\d+(?:[.,]\d+)?)\s*mm",
        t
    )

    if not diameter_match:

        diameter_match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*mm",
            t
        )

    if not diameter_match:
        return None

    diameter = parse_number(
        diameter_match.group(1)
    )

    if not diameter:
        return None

    # Jumlah batang
    batang_match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:batang|btg)",
        t
    )

    jumlah_batang = None

    if batang_match:
        jumlah_batang = int(
            parse_number(
                batang_match.group(1)
            )
        )

    # Panjang total
    panjang_match = re.search(
        r"(?:panjang|total)\s*"
        r"(\d+(?:[.,]\d+)?)\s*m",
        t
    )

    panjang_total = None

    if panjang_match:
        panjang_total = parse_number(
            panjang_match.group(1)
        )

    # Jika hanya "besi D10 20 m"
    # dan tidak ditemukan keyword panjang
    if panjang_total is None:

        simple_length = re.search(
            r"(\d+(?:[.,]\d+)?)\s*m",
            t
        )

        if simple_length:
            panjang_total = parse_number(
                simple_length.group(1)
            )

    standard_length = 12

    standard_match = re.search(
        r"(?:batang|panjang batang|standar)"
        r"\s*(?:=|:)?\s*"
        r"(\d+(?:[.,]\d+)?)\s*m",
        t
    )

    if standard_match:

        standard_length = parse_number(
            standard_match.group(1)
        )

    weight_per_meter = (
        diameter ** 2 / 162
    )

    if jumlah_batang is not None:

        panjang_total = (
            jumlah_batang * standard_length
        )

    if panjang_total is None:
        return None

    weight_total = (
        panjang_total *
        weight_per_meter
    )

    if jumlah_batang is None:

        jumlah_batang = math.ceil(
            panjang_total /
            standard_length
        )

    bought_length = (
        jumlah_batang *
        standard_length
    )

    return f"""
🔩 KEBUTUHAN BESI

📋 DATA

• Diameter: D{fmt(diameter)} mm
• Panjang total: {fmt(panjang_total)} m
• Panjang batang standar: {fmt(standard_length)} m
• Jumlah batang: {jumlah_batang}

🧮 PERHITUNGAN

Rumus berat besi:

D² / 162

= {fmt(diameter)}² / 162

= {fmt(weight_per_meter)} kg/m

Berat total:

{fmt(panjang_total)} × {fmt(weight_per_meter)}
= {fmt(weight_total)} kg

📊 RINGKASAN

• Besi: D{fmt(diameter)} mm
• Panjang total: {fmt(panjang_total)} m
• Jumlah batang: {jumlah_batang} batang
• Total panjang pembelian: {fmt(bought_length)} m
• Berat estimasi: {fmt(weight_total)} kg

🔍 VALIDASI

✅ Rumus berat menggunakan D² / 162
✅ Panjang pembelian berdasarkan batang standar

📝 CATATAN

Perhitungan ini menghitung berat dan kebutuhan material besi.
Ini bukan desain tulangan struktur.

Jumlah dan diameter tulangan struktur harus ditentukan
berdasarkan perhitungan engineer.
""".strip()


# ============================================================
# FOOTPLAT
# ============================================================

def civil_footplat_calculation(text):

    dims = extract_dimensions(text)

    if len(dims) < 3:
        return None

    p, l, t = dims[:3]

    volume_one = p * l * t

    qty_match = re.search(
        r"(\d+)\s*(?:buah|bh|unit|titik|buah footplat)",
        text.lower()
    )

    quantity = 1

    if qty_match:
        quantity = int(
            qty_match.group(1)
        )

    total_volume = (
        volume_one * quantity
    )

    materials = concrete_materials(
        total_volume
    )

    return f"""
🏗️ KEBUTUHAN FOOTPLAT

📋 DATA

• Panjang: {fmt(p)} m
• Lebar: {fmt(l)} m
• Tebal: {fmt(t)} m
• Jumlah: {quantity} buah

🧮 PERHITUNGAN

Volume 1 footplat:

{fmt(p)} × {fmt(l)} × {fmt(t)}
= {fmt(volume_one)} m³

Total beton:

{fmt(volume_one)} × {quantity}
= {fmt(total_volume)} m³

🔩 ESTIMASI MATERIAL BETON

• Semen: sekitar {materials["cement_sacks"]} zak
• Pasir: sekitar {fmt(materials["sand_m3"])} m³
• Kerikil: sekitar {fmt(materials["gravel_m3"])} m³
• Air: sekitar {fmt(materials["water_liter"])} liter

🔍 VALIDASI

• Volume sudah dikalikan jumlah footplat.
• Material menggunakan asumsi beton 1 : 2 : 3.

⚠️ CATATAN STRUKTURAL

Ukuran footplat dan tulangannya tidak boleh dianggap aman
hanya dari volume beton.

Diperlukan data seperti:
• beban
• kondisi tanah
• jumlah lantai
• ukuran kolom
• mutu beton
• mutu baja
• kondisi pondasi

Untuk desain final, verifikasi engineer struktur diperlukan.
""".strip()


# ============================================================
# SLOOF / KOLOM / BALOK / PLAT
# ============================================================

def civil_structural_member_calculation(
    text,
    member_name
):

    dims = extract_dimensions(text)

    if len(dims) < 3:
        return None

    a, b, length = dims[:3]

    quantity_match = re.search(
        r"(\d+)\s*(?:buah|bh|unit)",
        text.lower()
    )

    quantity = 1

    if quantity_match:
        quantity = int(
            quantity_match.group(1)
        )

    volume_one = (
        a * b * length
    )

    total_volume = (
        volume_one * quantity
    )

    materials = concrete_materials(
        total_volume
    )

    return f"""
🏗️ KEBUTUHAN {member_name.upper()}

📋 DATA

• Dimensi penampang: {fmt(a)} m × {fmt(b)} m
• Panjang: {fmt(length)} m
• Jumlah: {quantity}
• Volume 1 elemen: {fmt(volume_one)} m³
• Total volume: {fmt(total_volume)} m³

⚙️ ASUMSI BETON

• Campuran nominal: 1 : 2 : 3
• Faktor kering: 1,54
• Semen: 1.440 kg/m³
• 1 zak: 50 kg
• w/c sekitar 0,50

🧮 ESTIMASI MATERIAL

• Semen: sekitar {fmt(materials["cement_kg"])} kg
• Semen: sekitar {materials["cement_sacks"]} zak
• Pasir: sekitar {fmt(materials["sand_m3"])} m³
• Kerikil: sekitar {fmt(materials["gravel_m3"])} m³
• Air: sekitar {fmt(materials["water_liter"])} liter

🔍 VALIDASI

• Volume = luas penampang × panjang × jumlah
• Setiap angka menggunakan satuan meter.

⚠️ CATATAN STRUKTURAL

Perhitungan di atas adalah kebutuhan volume/material awal.

Belum menentukan apakah ukuran dan tulangan
{member_name.lower()} aman.

Untuk desain struktur diperlukan data beban,
bentang, mutu beton, mutu baja, kondisi tumpuan,
dan ketentuan desain yang berlaku.

🎯 KESIMPULAN

Volume beton:
{fmt(total_volume)} m³

Semen:
sekitar {materials["cement_sacks"]} zak
""".strip()


# ============================================================
# CIVIL MASTER ROUTER
# ============================================================

def civil_calculator(text):

    t = text.lower().strip()

    # --------------------------------------------------------
    # BESI
    # --------------------------------------------------------

    if (
        "besi" in t
        or re.search(r"\bd\d+\b", t)
        or "diameter" in t
    ):
        result = civil_rebar_calculation(text)

        if result:
            return result

    # --------------------------------------------------------
    # FOOTPLAT
    # --------------------------------------------------------

    if (
        "footplat" in t
        or "foot plate" in t
        or "tapak" in t
    ):
        result = civil_footplat_calculation(text)

        if result:
            return result

    # --------------------------------------------------------
    # SLOOF
    # --------------------------------------------------------

    if "sloof" in t:

        result = civil_structural_member_calculation(
            text,
            "sloof"
        )

        if result:
            return result

    # --------------------------------------------------------
    # KOLOM
    # --------------------------------------------------------

    if "kolom" in t:

        result = civil_structural_member_calculation(
            text,
            "kolom"
        )

        if result:
            return result

    # --------------------------------------------------------
    # BALOK
    # --------------------------------------------------------

    if "balok" in t:

        result = civil_structural_member_calculation(
            text,
            "balok"
        )

        if result:
            return result

    # --------------------------------------------------------
    # PLAT
    # --------------------------------------------------------

    if (
        "plat beton" in t
        or "plat lantai" in t
    ):

        result = civil_concrete_calculation(
            text,
            "KEBUTUHAN PLAT BETON"
        )

        if result:
            return result

    # --------------------------------------------------------
    # PONDASI BATU KALI
    # --------------------------------------------------------

    if (
        "pondasi batu kali" in t
        or "batu kali" in t
    ):

        dims = extract_dimensions(text)

        if len(dims) >= 3:

            p, l, h = dims[:3]

            volume = p * l * h

            stone = volume * 1.20
            mortar = volume * 0.20

            cement_kg = (
                mortar * 1.33 / 5 * 1440
            )

            cement_sacks = math.ceil(
                cement_kg / 50
            )

            sand = (
                mortar * 1.33 * 4 / 5
            )

            return f"""
🏗️ KEBUTUHAN PONDASI BATU KALI

📋 DATA

• Panjang: {fmt(p)} m
• Lebar rata-rata: {fmt(l)} m
• Tinggi: {fmt(h)} m
• Volume geometris: {fmt(volume)} m³

⚙️ ASUMSI

• Batu kali: sekitar 1,20 × volume geometris
• Mortar: sekitar 20% volume geometris
• Mortar 1 : 4
• Faktor kering mortar: 1,33
• Semen: 1.440 kg/m³
• 1 zak: 50 kg

🧮 ESTIMASI MATERIAL

• Batu kali: sekitar {fmt(stone)} m³
• Mortar: sekitar {fmt(mortar)} m³
• Semen: sekitar {fmt(cement_kg)} kg
• Semen: sekitar {cement_sacks} zak
• Pasir mortar: sekitar {fmt(sand)} m³

🔍 VALIDASI

• Volume geometris = panjang × lebar × tinggi
• Batu dan mortar menggunakan faktor estimasi.

⚠️ CATATAN

Ukuran pondasi tidak boleh ditentukan hanya berdasarkan
volume material.

Kondisi tanah, beban bangunan, kedalaman pondasi,
dan dimensi aktual harus diperiksa untuk desain final.
""".strip()

    # --------------------------------------------------------
    # DINDING
    # --------------------------------------------------------

    if (
        "dinding" in t
        or "tembok" in t
        or "bata" in t
        or "batako" in t
    ):

        result = civil_wall_calculation(text)

        if result:
            return result

    # --------------------------------------------------------
    # PLESTER
    # --------------------------------------------------------

    if "plester" in t:

        result = civil_plaster_calculation(text)

        if result:
            return result

    # --------------------------------------------------------
    # ACIAN
    # --------------------------------------------------------

    if "acian" in t:

        result = civil_acian_calculation(text)

        if result:
            return result

    # --------------------------------------------------------
    # GALIAN
    # --------------------------------------------------------

    if (
        "galian" in t
        or "menggali" in t
    ):

        result = civil_volume_calculation(
            text,
            "VOLUME GALIAN",
            "volume galian"
        )

        if result:
            return result

    # --------------------------------------------------------
    # URUGAN
    # --------------------------------------------------------

    if (
        "urugan" in t
        or "urug" in t
    ):

        result = civil_volume_calculation(
            text,
            "VOLUME URUGAN",
            "volume urugan"
        )

        if result:
            return result

    # --------------------------------------------------------
    # BETON / LANTAI
    # --------------------------------------------------------

    concrete_keywords = [
        "beton",
        "lantai beton",
        "cor",
        "ngecor",
        "coran",
        "sipil",
    ]

    if any(
        x in t
        for x in concrete_keywords
    ):

        result = civil_concrete_calculation(
            text
        )

        if result:
            return result

    return None


# ============================================================
# TASK CLASSIFIER
# ============================================================

