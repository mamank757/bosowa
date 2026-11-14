#!/usr/bin/env python3
"""
build_indices.py
Menggabungkan indeks iklim menjadi satu file bulanan: data/indices/monthly.json

  - IOD  : Dipole Mode Index (NOAA PSL, dmi.had.long.data), diunduh
  - MJO  : RMM (data/waves/rmm.txt yang sudah Anda tarik), diringkas per bulan
  - Kelvin / Rossby : file yang sudah ada di repo Anda (format dideteksi otomatis)

Kunci bulan memakai format 'YYYY-M' (mis. '2026-11') supaya sama dengan dashboard.

Pemakaian:
  python scripts/build_indices.py
  python scripts/build_indices.py --kelvin data/waves/kelvin.csv --rossby data/waves/rossby.csv --wave-stat std

Kelvin/Rossby: bila file harian (sinyal bertanda hasil filter OLR), ringkasan bulanan
yang dipakai adalah 'std' (aktivitas gelombang). Pilihan lain: 'mean', 'absmean'.
Bila file sudah bulanan, nilainya dipakai apa adanya.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

DMI_SOURCES = [   # (label, url, jenis, nama berkas cache). Urutan = prioritas.
    ("NOAA CPC DMI (ERSSTv6, klimatologi 1991-2020)",
     "https://www.cpc.ncep.noaa.gov/products/international/ocean_monitoring/indian/IODMI/mnth.ersstv6.clim19912020.dmi_current.txt",
     "cpc", "dmi_cpc.raw.txt"),
    ("NOAA PSL DMI (HadISST1.1)", "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data", "psl", "dmi_psl.raw.txt"),
]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
MJO_GROUPS = {"mc": [4, 5], "ioc": [2, 3], "wp": [6, 7], "wh": [8, 1]}


def warn(msg):
    print(f"::warning::{msg}", flush=True)


def info(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------- IOD (DMI)
def parse_dmi(text):
    """Tabel PSL: baris data = tahun + 12 nilai. Nilai hilang (-9999 dst.) dibuang."""
    out = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) != 13:
            continue
        try:
            y = int(p[0])
            vals = [float(x) for x in p[1:]]
        except ValueError:
            continue
        if not 1800 < y < 2200:
            continue
        for m, v in enumerate(vals, 1):
            if abs(v) < 50:
                out[f"{y}-{m}"] = round(v, 3)
    return out


def parse_dmi_cpc(text):
    """CPC: baris 'Tahun Bulan WTIO SETIO DMI'. Baris DMI divalidasi: DMI = WTIO - SETIO."""
    out = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        try:
            y, m = int(p[0]), int(p[1])
            w, st, d = float(p[2]), float(p[3]), float(p[4])
        except ValueError:
            continue
        if 1800 < y < 2200 and 1 <= m <= 12 and abs(d) < 50 and abs((w - st) - d) < 0.05:
            out[f"{y}-{m}"] = round(d, 3)
    return out


def _download(url, tries=3):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            last = e
            info(f"Unduh gagal ({url.split('/')[-1]}, percobaan {k + 1}): {e}")
            time.sleep(5)
    raise RuntimeError(str(last))


def load_dmi(raw_dir):
    """Urutan: CPC segar -> cache CPC -> PSL segar -> cache PSL. Mengembalikan (deret, label)."""
    parsers = {"cpc": parse_dmi_cpc, "psl": parse_dmi}
    for label, url, kind, fname in DMI_SOURCES:
        path = os.path.join(raw_dir, fname)
        try:
            txt = _download(url)
            d = parsers[kind](txt)
            if len(d) < 500:
                raise ValueError("format DMI tidak dikenali")
            os.makedirs(raw_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            return d, label
        except Exception as e:  # noqa: BLE001
            info(f"DMI {kind.upper()} segar gagal: {e}")
        if os.path.exists(path):
            d = parsers[kind](open(path, encoding="utf-8").read())
            if len(d) >= 500:
                warn(f"DMI: memakai salinan lama {fname} (unduhan gagal).")
                return d, label + " [salinan lama]"
    return None, None


# ----------------------------------------------------------------- MJO (RMM)
def parse_rmm(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) >= 7 and p[0].isdigit() and len(p[0]) == 4:
                rows.append(p[:7])
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=["year", "month", "day", "rmm1", "rmm2", "phase", "amp"]).astype(float)
    for c in ("rmm1", "rmm2", "amp"):
        d[c] = d[c].mask(d[c].abs() > 100)  # 1.E36 dan 999
    d["phase"] = d["phase"].mask(d["phase"] == 999)
    d = d.dropna(subset=["amp", "phase"])
    d["date"] = pd.to_datetime(d[["year", "month", "day"]].astype(int))
    return d


def mjo_monthly(d):
    d = d.copy()
    d["active"] = d["amp"] >= 1.0
    d["ym"] = d["date"].dt.year * 100 + d["date"].dt.month
    g = d.groupby("ym")
    res = {
        "amp": g["amp"].mean(),
        "active": g["active"].mean(),
        "rmm1": g["rmm1"].mean(),
        "rmm2": g["rmm2"].mean(),
    }
    for name, ph in MJO_GROUPS.items():
        res[name] = d.assign(x=d["active"] & d["phase"].isin(ph)).groupby("ym")["x"].mean()
    days = g.size()

    def key(ym):
        return f"{ym // 100}-{ym % 100}"

    out = {k: {key(i): round(float(v), 3) for i, v in s.items() if np.isfinite(v)} for k, s in res.items()}
    return out, {key(i): int(v) for i, v in days.items()}


# ------------------------------------------------------- Kelvin / Rossby
def find_file(name):
    """Cari file bernama mengandung 'kelvin'/'rossby' di folder data/."""
    for root, _dirs, files in os.walk("data"):
        for fn in sorted(files):
            low = fn.lower()
            if name in low and low.rsplit(".", 1)[-1] in ("csv", "txt", "dat", "tsv"):
                return os.path.join(root, fn)
    return None


def _is_num(x):
    try:
        float(str(x))
        return True
    except ValueError:
        return False


def read_flexible(path):
    """Baca tabel teks/CSV. Menangani: pemisah koma/titik-koma/tab/spasi, judul berupa
    baris komentar ('# year month day value'), dan file tanpa judul sama sekali."""
    comment_names = None
    with open(path, encoding="utf-8", errors="replace") as f:
        head = [next(f, "") for _ in range(8)]
    for ln in head:
        if ln.lstrip().startswith("#"):
            tok = ln.lstrip("# \t").replace(",", " ").replace(";", " ").split()
            if len(tok) >= 2 and not any(_is_num(t) for t in tok):
                comment_names = [t.lower() for t in tok]
    for kw in (dict(sep=None, engine="python", comment="#"),
               dict(sep=r"\s+", engine="python", comment="#")):
        try:
            df = pd.read_csv(path, **kw)
        except Exception:  # noqa: BLE001
            continue
        if df.shape[1] < 2 or len(df) <= 5:
            continue
        if all(_is_num(c) for c in df.columns):          # tidak ada baris judul: baris pertama adalah data
            df = pd.read_csv(path, header=None, **kw)
            n = df.shape[1]
            if comment_names and len(comment_names) == n:
                df.columns = comment_names
            elif n == 2:
                df.columns = ["date", "value"]
            elif n == 3:
                df.columns = ["year", "month", "value"]
            else:
                df.columns = ["year", "month", "day"] + [f"v{i}" for i in range(1, n - 2)]
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    return None


def detect_dates(df):
    cols = list(df.columns)
    for c in ("date", "tanggal", "time", "waktu", "datetime", "tgl"):
        if c in cols:
            t = pd.to_datetime(df[c], errors="coerce")
            if t.notna().mean() > 0.8:
                return t, [c]
    for ys, ms, ds in (("year", "month", "day"), ("tahun", "bulan", "hari"), ("yr", "mon", "dy")):
        if ys in cols and ms in cols:
            parts = {"year": df[ys], "month": df[ms], "day": df[ds] if ds in cols else 15}
            return pd.to_datetime(parts, errors="coerce"), [ys, ms] + ([ds] if ds in cols else [])
    return None, []


def load_wave(path, name, stat):
    if not path or not os.path.exists(path):
        warn(f"{name}: file tidak ditemukan (isi --{name} atau taruh file bernama '{name}*.csv' di folder data/).")
        return None, None
    df = read_flexible(path)
    if df is None:
        warn(f"{name}: {path} tidak bisa dibaca sebagai tabel.")
        return None, None
    t, used = detect_dates(df)
    if t is None:
        warn(f"{name}: kolom tanggal tidak terdeteksi. Kolom: {list(df.columns)}")
        return None, None
    cand = [c for c in df.columns if c not in used and pd.api.types.is_numeric_dtype(df[c])]
    pick = next((c for c in cand if name in c), None)
    if pick is None:
        pick = next((c for c in cand if any(k in c for k in ("value", "amp", "var", "olr", "anom", "index"))), None)
    if pick is None and cand:
        pick = cand[-1]
    if pick is None:
        warn(f"{name}: kolom nilai tidak ditemukan. Kolom: {list(df.columns)}")
        return None, None
    v = pd.to_numeric(df[pick], errors="coerce")
    v = v.mask(v.abs() > 1e5)
    ok = t.notna() & v.notna()
    s = pd.DataFrame({"t": t[ok], "v": v[ok]})
    s["ym"] = s["t"].dt.year * 100 + s["t"].dt.month
    g = s.groupby("ym")["v"]
    n = g.size()
    if n.median() < 3:  # data sudah bulanan
        res, how = g.mean(), "bulanan apa adanya"
    elif stat == "mean":
        res, how = g.mean(), "rata-rata bulanan"
    elif stat == "absmean":
        res, how = s.assign(v=s["v"].abs()).groupby("ym")["v"].mean(), "rata-rata |nilai|"
    else:
        res, how = g.std(ddof=0), "simpangan baku bulanan"

    def key(ym):
        return f"{ym // 100}-{ym % 100}"

    info(f"{name}: {path} | kolom '{pick}' | {how} | {len(res)} bulan")
    return ({key(i): round(float(x), 4) for i, x in res.items() if np.isfinite(x)},
            {key(i): int(x) for i, x in n.items()})


# ----------------------------------------------------------------- utama
def trim(m, since):
    return {k: v for k, v in m.items() if int(k.split("-")[0]) >= since}


def span(m):
    ks = sorted(m, key=lambda k: tuple(map(int, k.split("-"))))
    return f"{ks[0]} .. {ks[-1]} ({len(ks)})" if ks else "kosong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmm", default="data/waves/rmm.txt")
    ap.add_argument("--kelvin", default=None)
    ap.add_argument("--rossby", default=None)
    ap.add_argument("--wave-stat", default="std", choices=["std", "mean", "absmean"])
    ap.add_argument("--dmi-dir", default="data/indices", help="folder cache berkas mentah DMI")
    ap.add_argument("--rmm-url", default=None, help="cadangan: unduh rmm.txt dari URL ini bila berkas lokal tidak ada")
    ap.add_argument("--since", type=int, default=1990)
    ap.add_argument("--out", default="data/indices/monthly.json")
    a = ap.parse_args()

    out = {"generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "days": {}}

    dmi, dmi_label = load_dmi(a.dmi_dir)
    if dmi:
        out["dmi"] = trim(dmi, a.since)
        out["dmi_source"] = dmi_label
        info(f"DMI [{dmi_label}]: {span(out['dmi'])}")
    else:
        warn("DMI tidak tersedia.")

    if not os.path.exists(a.rmm) and a.rmm_url:
        try:
            os.makedirs(os.path.dirname(a.rmm) or ".", exist_ok=True)
            with open(a.rmm, "w", encoding="utf-8") as f:
                f.write(_download(a.rmm_url))
            info(f"RMM: {a.rmm} tidak ada, diunduh dari {a.rmm_url}")
        except Exception as e:  # noqa: BLE001
            warn(f"RMM: unduhan cadangan gagal: {e}")

    if os.path.exists(a.rmm):
        d = parse_rmm(a.rmm)
        if d is not None:
            mj, days = mjo_monthly(d)
            out["mjo"] = {k: trim(v, a.since) for k, v in mj.items()}
            out["days"]["mjo"] = trim(days, a.since)
            info(f"MJO: {span(out['mjo']['mc'])}")
        else:
            warn("RMM: tidak ada baris data terbaca.")
    else:
        warn(f"RMM: {a.rmm} tidak ada.")

    for name, path in (("kelvin", a.kelvin), ("rossby", a.rossby)):
        v, n = load_wave(path or find_file(name), name, a.wave_stat)
        if v:
            out[name] = trim(v, a.since)
            out["days"][name] = trim(n, a.since)

    if not any(k in out for k in ("dmi", "mjo", "kelvin", "rossby")):
        print("Tidak ada indeks yang berhasil dibuat.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), ensure_ascii=False)
    info(f"Ditulis {a.out} ({os.path.getsize(a.out) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
