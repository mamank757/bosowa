#!/usr/bin/env python3
"""Unduh Niño-3.4 relatif mingguan dari BoM -> data.json (rata-rata bulanan).

Format sumber (tanpa header):  mulai,akhir,nilai   mis. 20261026,20261101,2.88
Tiap minggu dibagi per HARI ke bulan yang ditempatinya (minggu yang melintasi
pergantian bulan dihitung proporsional), lalu dirata-ratakan per bulan.

Pemakaian:  python fetch_bom.py            (unduh dari BoM)
            python fetch_bom.py file.txt   (uji dengan file lokal)
"""
import calendar, json, sys, time, urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

URL = "https://www.bom.gov.au/clim_data/IDCK000072/rnino_3.4.txt"
OUT = Path(__file__).resolve().parent / "data.json"   # data.json di folder yang sama (root repo)
MIN_HARI = 14      # bulan dipakai bila minimal 14 hari tercakup (kurang dari itu dibuang)
MAKS = 5           # |nilai| > 5 dianggap rusak
MIN_MINGGU = 300   # pengaman: file yang terbaca terlalu sedikit = format berubah -> gagal


def unduh():
    galat = None
    for i in range(4):
        try:
            req = urllib.request.Request(
                f"{URL}?{int(time.time() * 1000)}",
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
                    "Accept": "text/plain,*/*;q=0.8",
                    "Referer": "https://www.bom.gov.au/climate/enso/",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            galat = e
            time.sleep(5 * (i + 1))
    sys.exit(f"Gagal mengunduh BoM: {galat}")


def parse(teks):
    minggu = []
    for baris in teks.splitlines():
        p = baris.strip().split(",")
        if len(p) != 3:
            continue
        try:
            a = datetime.strptime(p[0], "%Y%m%d").date()
            b = datetime.strptime(p[1], "%Y%m%d").date()
            v = float(p[2])
        except ValueError:
            continue
        if b < a or abs(v) > MAKS:
            continue
        minggu.append((a, b, v))
    if len(minggu) < MIN_MINGGU:
        sys.exit(f"Format BoM tidak dikenali ({len(minggu)} minggu terbaca)")
    return minggu


def bulanan(minggu):
    jml, n = defaultdict(float), defaultdict(int)
    for a, b, v in minggu:
        d = a
        while d <= b:
            jml[(d.year, d.month)] += v
            n[(d.year, d.month)] += 1
            d += timedelta(days=1)
    data, sementara = {}, []
    for (y, m) in sorted(n):
        dim = calendar.monthrange(y, m)[1]
        if n[(y, m)] < min(MIN_HARI, dim):
            continue
        data.setdefault(str(y), [None] * 12)[m - 1] = round(jml[(y, m)] / n[(y, m)], 2)
        if n[(y, m)] < dim:
            sementara.append(f"{y}-{m:02d}")
    terakhir = max((f"{y}-{m:02d}" for y, m in n if f"{y}" in data and data[f"{y}"][m - 1] is not None), default="")
    return data, sementara, terakhir


def main():
    teks = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else unduh()
    minggu = parse(teks)
    data, sementara, terakhir = bulanan(minggu)
    hasil = {
        "sumber": URL,
        "keterangan": "Anomali Niño-3.4 relatif (BoM), rata-rata bulanan dari data mingguan",
        "minggu_terakhir": max(b for _, b, _ in minggu).isoformat(),
        "terakhir": terakhir,
        "sementara": sementara,
        "data": data,
    }
    baru = json.dumps(hasil, ensure_ascii=False, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() == baru:
        print("Tidak ada perubahan.")
        return
    OUT.write_text(baru)
    print(f"data.json ditulis: {len(minggu)} minggu -> terakhir {terakhir}, sementara {sementara}")


if __name__ == "__main__":
    main()
