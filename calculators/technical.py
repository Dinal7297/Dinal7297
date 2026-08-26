
import math

class TechnicalCalculator:
    DENSITY = {
        "steel": 7850, "baja": 7850, "besi": 7850,
        "stainless": 7930, "aluminum": 2700, "alumunium": 2700
    }

    @staticmethod
    def calculate_plate_weight(length_m: float, width_m: float, thickness_mm: float, material: str = "steel") -> dict:
        thickness_m = thickness_mm / 1000.0
        volume_m3 = length_m * width_m * thickness_m
        rho = TechnicalCalculator.DENSITY.get(material.lower(), 7850)
        weight_kg = volume_m3 * rho
        return {
            "tipe": "Plat Logam",
            "dimensi": f"{length_m}m x {width_m}m x {thickness_mm}mm",
            "volume_m3": round(volume_m3, 6),
            "weight_kg": round(weight_kg, 2),
            "material": material.capitalize()
        }

    @staticmethod
    def calculate_hollow_weight(length_m: float, width_mm: float, height_mm: float, thickness_mm: float, material: str = "steel") -> dict:
        w_m, h_m, t_m = width_mm / 1000.0, height_mm / 1000.0, thickness_mm / 1000.0
        area_outer = w_m * h_m
        area_inner = max(0.0, (w_m - 2*t_m) * (h_m - 2*t_m))
        cross_area_m2 = area_outer - area_inner
        volume_m3 = cross_area_m2 * length_m
        rho = TechnicalCalculator.DENSITY.get(material.lower(), 7850)
        return {
            "tipe": "Pipa Hollow",
            "dimensi": f"{width_mm}x{height_mm}mm, Tebal {thickness_mm}mm, Panjang {length_m}m",
            "weight_kg": round(volume_m3 * rho, 2)
        }

    @staticmethod
    def estimate_welding_consumable(weld_length_m: float, fillet_size_mm: float, process: str = "SMAW") -> dict:
        leg_m = fillet_size_mm / 1000.0
        weld_volume_m3 = 0.5 * (leg_m ** 2) * weld_length_m
        metal_weight_kg = weld_volume_m3 * 7850
        efficiency = 0.55 if process.upper() == "SMAW" else 0.85
        required_wire_kg = metal_weight_kg / efficiency
        return {
            "tipe": "Estimasi Bahan Las",
            "panjang_las_m": weld_length_m,
            "ukuran_fillet_mm": fillet_size_mm,
            "proses": process.upper(),
            "berat_deposit_kg": round(metal_weight_kg, 2),
            "estimasi_kawat_kg": round(required_wire_kg, 2)
        }

    @staticmethod
    def convert_unit(value: float, from_unit: str, to_unit: str) -> dict:
        units = {
            ("mm", "m"): value / 1000.0, ("m", "mm"): value * 1000.0,
            ("cm", "mm"): value * 10.0, ("mm", "cm"): value / 10.0,
            ("inch", "mm"): value * 25.4, ("mm", "inch"): value / 25.4,
            ("kg", "ton"): value / 1000.0, ("ton", "kg"): value * 1000.0,
            ("mpa", "kgf_cm2"): value * 10.1972, ("kgf_cm2", "mpa"): value / 10.1972
        }
        res = units.get((from_unit.lower(), to_unit.lower()), None)
        return {
            "tipe": "Konversi Satuan",
            "input": f"{value} {from_unit}",
            "hasil": round(res, 4) if res is not None else "Satuan tidak didukung",
            "to_unit": to_unit
        }


