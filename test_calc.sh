#!/bin/sh

echo "=== TEST TECHNICAL CALCULATOR V1 (NATIVE ENGINE) ==="

# 1. Plat Baja: panjang 2m, lebar 1m, tebal 3mm (0.003m), density 7850
L=2; W=1; T=0.003; RHO=7850
VOL=$(awk "BEGIN {print $L * $W * $T}")
BERAT=$(awk "BEGIN {print $VOL * $RHO}")
echo "\n[1] Plat Baja 2m x 1m x 3mm:"
echo "    Volume : $VOL m3"
echo "    Berat  : $BERAT kg"

# 2. Las Fillet 6mm (0.006m) panjang 12m (SMAW eff 0.55)
W_LEN=12; FILLET=0.006
VOL_LAS=$(awk "BEGIN {print 0.5 * ($FILLET ^ 2) * $W_LEN}")
DEPOSIT=$(awk "BEGIN {print $VOL_LAS * 7850}")
ELEKTRODA=$(awk "BEGIN {print $DEPOSIT / 0.55}")
echo "\n[2] Kebutuhan Las Fillet 6mm x 12m (SMAW):"
echo "    Deposit Metal : $DEPOSIT kg"
echo "    Est. Elektroda: $ELEKTRODA kg"

# 3. Konversi 50 MPa ke kgf/cm2
MPA=50
KGF=$(awk "BEGIN {print $MPA * 10.1972}")
echo "\n[3] Konversi 50 MPa ke kgf/cm2:"
echo "    Hasil: $KGF kgf/cm2"

