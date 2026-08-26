from calculators.technical import TechnicalCalculator

print("=== TEST TECHNICAL CALCULATOR V1 ===")

# Test 1: Hitung Berat Plat
plat = TechnicalCalculator.calculate_plate_weight(length_m=2.0, width_m=1.0, thickness_mm=3.0, material="steel")
print("\n[1] Plat Baja 2m x 1m x 3mm:")
print(f"    Volume : {plat['volume_m3']} m3")
print(f"    Berat  : {plat['weight_kg']} kg")

# Test 2: Estimasi Kawat Las SMAW
las = TechnicalCalculator.estimate_welding_consumable(weld_length_m=12.0, fillet_size_mm=6.0, process="SMAW")
print("\n[2] Kebutuhan Las Fillet 6mm x 12m (SMAW):")
print(f"    Deposit Metal : {las['berat_deposit_kg']} kg")
print(f"    Est. Elektroda: {las['estimasi_kawat_kg']} kg")

# Test 3: Konversi Satuan
konv = TechnicalCalculator.convert_unit(value=50, from_unit="MPa", to_unit="kgf_cm2")
print("\n[3] Konversi 50 MPa ke kgf/cm2:")
print(f"    Hasil: {konv['hasil']} {konv['to_unit']}")

