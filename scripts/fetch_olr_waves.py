#!/usr/bin/env python3
"""
fetch_olr_waves.py
Menghitung deret harian gelombang Kelvin dan Rossby ekuator (ER n=1) dari OLR harian NOAA,
lalu menulis data/waves/kelvin.csv dan data/waves/rossby.csv.

Mengapa dihitung sendiri: NOAA/BoM hanya menerbitkan peta dan grafik gelombang ini, bukan
deret angka siap unduh. Metodenya mengikuti Wheeler & Kiladis (1999) dan
Wheeler & Weickmann (2001), dengan batas filter seperti Kiladis dkk. (2009):

  Kelvin : eastward,  wavenumber 1-14, periode 2.5-17 hari, kedalaman ekuivalen 8-90 m
  ER n=1 : westward,  wavenumber 1-10, periode 9-72 hari,  di bawah kurva h = 90 m

Langkah:
  1. Unduh OLR via OPeNDAP (pita |lintang| <= 10, longitude tiap ~3 derajat), per tahun, dengan cache.
  2. Komponen simetris terhadap ekuator, dirata-rata di pita lintang -> deret (waktu x longitude).
  3. Anomali: buang rata-rata dan 3 harmonik musiman.
  4. Filter bilangan-gelombang-frekuensi 2D (FFT), zero-padding di ujung seperti metode waktu-nyata
     Wheeler & Weickmann (2001).
  5. Rata-rata di kotak wilayah (default 115-125E) -> CSV harian (W/m2, negatif = konveksi menguat).

Nilai ~hari-hari terakhir bersifat sementara (efek ujung deret); --trim-end membuangnya.
Ringkasan bulanan (simpangan baku) dibuat oleh build_indices.py.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

A_EARTH = 6.371e6
OMEGA = 7.292e-5
G = 9.8
DEFAULT_URLS = [
    "https://psl.noaa.gov/thredds/dodsC/Datasets/olr_cdr_day/olr.day.mean.v2.nc",
    "https://psl.noaa.gov/thredds/dodsC/Datasets/cpc_blended_olr-1deg/olr.day.mean.nc",
]


def info(m):
    print(m, flush=True)


def warn(m):
    print(f"::warning::{m}", flush=True)


# ------------------------------------------------------------------ filter
def f_kelvin(h, s):
    """Frekuensi (siklus/hari) gelombang Kelvin pada kedalaman ekuivalen h (m), wavenumber s."""
    return s * np.sqrt(G * h) * 86400.0 / (2 * np.pi * A_EARTH)


def f_er(h, s):
    """Frekuensi (siklus/hari) ER n=1 pada kedalaman ekuivalen h (m), wavenumber |s|."""
    k = s / A_EARTH
    c = np.sqrt(G * h)
    beta = 2 * OMEGA / A_EARTH
    w = beta * k / (k ** 2 + 3 * beta / c)
    return w * 86400.0 / (2 * np.pi)


def band_masks(nt, nlon):
    """Topeng Kelvin dan ER pada grid FFT 2D (waktu x longitude)."""
    s = np.fft.fftfreq(nlon, d=1.0 / nlon)         # wavenumber bulat (siklus mengelilingi bumi)
    fi = np.fft.fftfreq(nt, d=1.0)                  # siklus/hari, tanda numpy
    S, F = np.meshgrid(s, fi)                       # (nt, nlon)
    fp = -F                                         # numpy: e^{-i2pi f t} -> eastward <=> S*fp > 0
    aS, aF = np.abs(S), np.abs(fp)
    east, west = (S * fp) > 0, (S * fp) < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        kel = (east & (aS >= 1) & (aS <= 14) & (aF >= 1 / 17.0) & (aF <= 1 / 2.5)
               & (aF >= f_kelvin(8, aS)) & (aF <= f_kelvin(90, aS)))
        er = west & (aS >= 1) & (aS <= 10) & (aF >= 1 / 72.0) & (aF <= 1 / 9.0) & (aF <= f_er(90, aS))
    return kel, er


def anomalies(z, dates):
    """Buang rata-rata + 3 harmonik musiman per longitude. NaN diisi interpolasi (maks 15 hari)."""
    df = pd.DataFrame(z, index=dates).interpolate(limit=15, limit_area="inside")
    df = df.fillna(df.mean())
    v = df.values.astype(np.float64)
    d = (dates.dayofyear.values - 1) / 365.25
    cols = [np.ones_like(d)]
    for k in (1, 2, 3):
        cols += [np.cos(2 * np.pi * k * d), np.sin(2 * np.pi * k * d)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    out = v - X @ beta
    return np.nan_to_num(out, nan=0.0)


def wave_filter(anom, pad=128):
    """Terapkan filter Kelvin dan ER. anom: (nt, nlon), longitude menaik. Mengembalikan (kelvin, er)."""
    nt, nlon = anom.shape
    n = nt + 2 * pad
    zp = np.zeros((n, nlon))
    zp[pad:pad + nt] = anom
    spec = np.fft.fft2(zp)
    kel, er = band_masks(n, nlon)
    kf = np.fft.ifft2(spec * kel).real[pad:pad + nt]
    ef = np.fft.ifft2(spec * er).real[pad:pad + nt]
    return kf, ef


# ------------------------------------------------------------------ data OLR
def open_dataset(urls, max_age_days):
    import xarray as xr
    last_err = None
    for url in urls:
        for engine in ("netcdf4", "pydap", None):
            try:
                kw = {"engine": engine} if engine else {}
                ds = xr.open_dataset(url, **kw)
                tn = "time" if "time" in ds.coords else next(c for c in ds.coords if "time" in c.lower())
                t_last = pd.Timestamp(ds[tn].values[-1])
                age = (pd.Timestamp.now("UTC").tz_localize(None) - t_last).days
                if age > max_age_days:
                    warn(f"{url}: data terakhir {t_last.date()} ({age} hari lalu), dilewati.")
                    ds.close()
                    break
                info(f"Sumber OLR: {url} (engine={engine or 'auto'}) | data terakhir {t_last.date()}")
                return ds, url
            except Exception as e:  # noqa: BLE001
                last_err = f"{url} [{engine}]: {e}"
    raise RuntimeError(f"Tidak ada sumber OLR yang bisa dibuka/segar. Galat terakhir: {last_err}")


def names(ds):
    def pick(cands):
        for c in cands:
            if c in ds.coords or c in ds.variables:
                return c
        raise KeyError(f"koordinat {cands} tidak ada: {list(ds.coords)}")
    var = "olr" if "olr" in ds.data_vars else next(v for v in ds.data_vars if ds[v].ndim == 3)
    return var, pick(["time"]), pick(["lat", "latitude"]), pick(["lon", "longitude"])


def band_pairs(lat, band):
    """Pasangan indeks (belahan selatan, utara) yang simetris terhadap ekuator, |lintang| <= band."""
    pairs = []
    for i, t in enumerate(lat):
        if -band - 1e-6 <= t <= 1e-6:
            j = int(np.argmin(np.abs(lat + t)))
            if abs(lat[j] + t) < 0.01:
                pairs.append((i, j))
    if not pairs:
        raise RuntimeError("Tidak ada pasangan lintang simetris di pita yang diminta.")
    return pairs


def load_chunk(ds, names_, i0, i1, lat_slice, lon_idx, pairs_local, retries=4):
    var, tn, ln, gn = names_
    for k in range(retries):
        try:
            blk = ds[var].isel({tn: slice(i0, i1), ln: lat_slice, gn: lon_idx}).values.astype(np.float32)
            blk[(blk < 20) | (blk > 500)] = np.nan          # OLR fisis ~ 50-400 W/m2
            parts = [np.nanmean(np.stack([blk[:, a, :], blk[:, b, :]]), axis=0) for a, b in pairs_local]
            return np.nanmean(np.stack(parts), axis=0)      # (t, lon)
        except Exception as e:  # noqa: BLE001
            info(f"  percobaan {k + 1}/{retries} gagal: {e}")
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"Gagal mengunduh blok waktu {i0}:{i1}")


def build_series(ds, args):
    names_ = names(ds)
    var, tn, ln, gn = names_
    times = pd.DatetimeIndex(ds[tn].values).normalize()
    lat = np.asarray(ds[ln].values, dtype=float)
    lon = np.asarray(ds[gn].values, dtype=float) % 360.0
    # lintang: potong blok kontigu yang mencakup pita
    sel = np.where(np.abs(lat) <= args.band + 1e-6)[0]
    lat_slice = slice(int(sel.min()), int(sel.max()) + 1)
    lat_loc = lat[lat_slice]
    pairs = band_pairs(lat_loc, args.band)
    # longitude: jarak ~3 derajat, urut menaik
    dlon = float(np.median(np.diff(np.sort(lon))))
    stride = max(1, int(round(args.lon_step / dlon)))
    order = np.argsort(lon)
    lon_idx = order[::stride]
    lon_sel = lon[lon_idx]
    if abs(360.0 / len(lon_sel) - np.median(np.diff(lon_sel))) > 0.5:
        raise RuntimeError("Grid longitude tidak seragam mengelilingi bumi.")
    info(f"Grid: {len(lat_loc)} lintang dipakai ({len(pairs)} pasangan simetris), {len(lon_sel)} longitude (tiap {dlon * stride:.1f} derajat)")

    # cache: dates + z (t, lon)
    z_old, d_old = None, None
    if args.cache and os.path.exists(args.cache):
        c = np.load(args.cache, allow_pickle=False)
        if c["lon"].shape == lon_sel.shape and np.allclose(c["lon"], lon_sel):
            d_old = pd.DatetimeIndex(c["dates"].astype("datetime64[ns]"))
            z_old = c["z"]
            info(f"Cache dimuat: {d_old[0].date()} .. {d_old[-1].date()} ({len(d_old)} hari)")
        else:
            warn("Cache tidak cocok dengan grid, diabaikan.")
    start = pd.Timestamp(args.start)
    if d_old is not None:
        start = max(start, d_old[-1] - pd.Timedelta(days=args.refetch_days))
        start = min(start, d_old[-1])
    i_start = int(np.searchsorted(times.values, start.to_datetime64()))
    i_end = len(times)
    info(f"Mengunduh indeks waktu {i_start}..{i_end} ({times[i_start].date()} .. {times[-1].date()})")
    chunks, dates_new = [], []
    step = 366
    for a in range(i_start, i_end, step):
        b = min(a + step, i_end)
        z = load_chunk(ds, names_, a, b, lat_slice, lon_idx, pairs)
        chunks.append(z)
        dates_new.append(times[a:b])
        info(f"  {times[a].date()} .. {times[b - 1].date()}  ok")
    z_new = np.concatenate(chunks) if chunks else np.zeros((0, len(lon_sel)), np.float32)
    d_new = dates_new[0].append(dates_new[1:]) if dates_new else pd.DatetimeIndex([])

    if d_old is not None and len(d_old):
        keep = d_old < d_new[0] if len(d_new) else np.ones(len(d_old), bool)
        z_all = np.concatenate([z_old[keep], z_new])
        d_all = d_old[keep].append(d_new)
    else:
        z_all, d_all = z_new, d_new
    # cache baru
    if args.cache:
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        np.savez_compressed(args.cache, dates=d_all.values.astype("datetime64[D]"), z=z_all.astype(np.float32), lon=lon_sel)
    return d_all, z_all, lon_sel


def compute(d_all, z_all, lon_sel, args):
    full = pd.date_range(d_all[0], d_all[-1], freq="D")
    z = pd.DataFrame(z_all, index=d_all)
    z = z[~z.index.duplicated(keep="last")].reindex(full)
    miss = float(np.isnan(z.values).mean())
    if miss > 0.02:
        warn(f"{miss * 100:.1f}% sel OLR kosong (diisi interpolasi).")
    anom = anomalies(z.values, full)
    kf, ef = wave_filter(anom)
    lo, hi = args.lon_min % 360, args.lon_max % 360
    m = (lon_sel >= lo) & (lon_sel <= hi) if lo <= hi else (lon_sel >= lo) | (lon_sel <= hi)
    if not m.any():
        raise RuntimeError("Tidak ada longitude di kotak wilayah.")
    return full, kf[:, m].mean(axis=1), ef[:, m].mean(axis=1), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", help="URL OPeNDAP/berkas netCDF OLR harian (boleh diulang)")
    ap.add_argument("--start", default="1996-01-01", help="awal data yang diunduh (untuk spin-up)")
    ap.add_argument("--out-start", default="1997-01-01", help="awal data yang ditulis ke CSV")
    ap.add_argument("--band", type=float, default=10.0, help="pita |lintang| (derajat) untuk komponen simetris")
    ap.add_argument("--lon-step", type=float, default=3.0)
    ap.add_argument("--lon-min", type=float, default=115.0)
    ap.add_argument("--lon-max", type=float, default=125.0)
    ap.add_argument("--trim-end", type=int, default=7, help="buang N hari terakhir (efek ujung deret)")
    ap.add_argument("--refetch-days", type=int, default=30, help="unduh ulang N hari terakhir dari cache")
    ap.add_argument("--max-age-days", type=int, default=20, help="tolak sumber yang datanya lebih tua dari ini")
    ap.add_argument("--cache", default=".cache/olr_strip.npz")
    ap.add_argument("--out-dir", default="data/waves")
    a = ap.parse_args()

    ds, url = open_dataset(a.url or DEFAULT_URLS, a.max_age_days)
    d_all, z_all, lon_sel = build_series(ds, a)
    full, kel, er, nlon_box = compute(d_all, z_all, lon_sel, a)

    os.makedirs(a.out_dir, exist_ok=True)
    keep = (full >= pd.Timestamp(a.out_start)) & (full <= full[-1] - pd.Timedelta(days=a.trim_end))
    for name, v in (("kelvin", kel), ("rossby", er)):
        pd.DataFrame({"date": full[keep].strftime("%Y-%m-%d"), name: np.round(v[keep], 3)}).to_csv(
            os.path.join(a.out_dir, f"{name}.csv"), index=False)
    meta = {"generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": url,
            "olr_last_date": str(full[-1].date()), "written_last_date": str(full[keep][-1].date()),
            "box": {"lon": [a.lon_min, a.lon_max], "lat_band_symmetric": a.band, "grid_points_lon": nlon_box},
            "kelvin": "eastward k=1-14, 2.5-17 d, h=8-90 m", "rossby": "ER n=1 westward k=1-10, 9-72 d, h<=90 m",
            "units": "W/m2 (filtered OLR anomaly; negative = enhanced convection)"}
    json.dump(meta, open(os.path.join(a.out_dir, "waves_meta.json"), "w"), indent=1)
    info(f"Ditulis kelvin.csv dan rossby.csv sampai {full[keep][-1].date()} ({keep.sum()} hari)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"::error::{e}", file=sys.stderr)
        sys.exit(1)
