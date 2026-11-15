/* prediktor.js — muat SETELAH perbaikan.js (dua baris sebelum </body>):
     <script src="perbaikan.js"></script>
     <script src="prediktor.js"></script>

   Prediksi bulanan dan tahunan = Baseline + ENSO + IOD + MJO + Kelvin + Rossby.

   Cara kerja (per kabupaten x bulan kalender):
   1. Model log-link (Poisson/IRLS) dengan penalti ridge pada prediktor terstandar.
   2. Rantai model bersarang: Baseline -> +ENSO -> +IOD -> +MJO -> +Kelvin -> +Rossby.
   3. Tiap model dinilai dengan leave-one-year-out (LOYO). Model yang lebih kompleks hanya
      dipakai bila menurunkan MSE minimal MARGIN (3%). Bila tidak ada yang mengalahkan
      klimatologi, hasilnya = Baseline.
   4. Prediktor yang datanya belum ada di monthly.json otomatis dilewati.
   5. Prediksi tahunan = jumlah prediksi bulanan; ketidakpastiannya dari galat LOYO tahunan.

   TIGA BULANAN (untuk 6-12 bulan ke depan): model dilatih per jarak prakiraan. Prediktor ENSO/IOD adalah
   rata-rata 3 bulan terakhir yang SUDAH teramati, dengan jeda yang sama persis pada data latih, sehingga
   tidak ada ekstrapolasi ENSO. Model BULANAN (lag 1-3 bulan) hanya dipakai sampai MONTHLY_MAX_LEAD bulan.

   Catatan penting: MJO/Kelvin/Rossby memakai lag 1 bulan. Untuk bulan yang prediktor
   lag-1 nya belum teramati, nilainya dianggap netral (tidak berpengaruh). */
(function () {
  'use strict';

  const C = {
    URL: 'https://raw.githubusercontent.com/mamank757/bosowa/refs/heads/main/data/indices/monthly.json',
    LS: 'ch_idx_v1',
    LAMBDA: 8,          // penalti ridge (satuan prediktor terstandar; setara SHRINK pada kode lama)
    MARGIN: 0.03,       // dan MSE LOYO harus turun >= MARGIN (relatif)
    ZSEL: 1.5,          // selisih galat kuadrat per tahun (LOYO) harus >= ZSEL x galat baku (uji berpasangan)
    TAU_IOD: 2,         // peluruhan IOD (bulan) untuk bulan yang belum teramati
    ZMAX: 2.5,          // batas |z| prediktor terstandar
    MJO_VAR: 'mc',      // fraksi hari MJO aktif di fase 4-5 (Maritime Continent)
    MIN_DAYS: 20,       // bulan harian dianggap lengkap bila >= 20 hari
    MIN_DAYS_NOW: 10,   // bulan berjalan boleh dipakai bila >= 10 hari
    MIN_COVER: 0.6,     // prediktor dipakai bila tersedia pada >= 60% tahun latih
    MONTHLY_MAX_LEAD: 3, // model BULANAN (lag 1-3 bulan) hanya dipakai sampai lead ini; sesudahnya Baseline (pakai tabel 3 bulanan)
    SEAS_MAX_LAG: 36,    // prediktor 3 bulanan diabaikan bila data terakhirnya lebih tua dari ini (bulan)
    POS_MIN: 0           // bulan dengan rata-rata pos aktif < nilai ini tidak dipakai sebagai data latih (0 = pakai semua)
  };
  const PRED = [
    { id: 'enso', nm: 'ENSO' },
    { id: 'iod', nm: 'IOD' },
    { id: 'mjo', nm: 'MJO' },
    { id: 'kelvin', nm: 'Kelvin' },
    { id: 'rossby', nm: 'Rossby' }
  ];
  let IDX = null, iodC = null;
  { const v = bacaLS('ch_posmin_v1'); if (typeof v === 'number' && isFinite(v) && v >= 0) C.POS_MIN = v; }
  /* Filter jumlah pos hanya untuk data STASIUN. Data areal (jumlah pos = 1 tetap, mis. rata-rata grid dari GEE/CHIRPS)
     tidak punya konsep jumlah pos, jadi filter diabaikan agar tidak membuang seluruh data. */
  function isArealK(id) {
    const c = A[id].fit;
    if (c._areal === undefined) { const st = posStat(id); c._areal = st.length > 0 && st.every(x => x.pos <= 1.0001); }
    return c._areal;
  }
  function okPos(id, t) {
    if (!(C.POS_MIN > 0) || isArealK(id)) return true;
    const r = (A[id].years[t.y] || [])[t.mo - 1];
    return !r || r.pos_rata == null || r.pos_rata >= C.POS_MIN;
  }

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const sum = a => a.reduce((s, x) => s + x, 0);
  const mean = a => sum(a) / a.length;
  const sg2 = v => v == null ? '–' : (v > 0 ? '+' : '') + fmt(v, 2);
  const sgm = v => v == null ? '–' : (v > 0 ? '+' : '') + fmt(v, 0);

  /* ================= DATA INDEKS ================= */
  function serie(pid) {
    if (!IDX) return null;
    if (pid === 'iod') return IDX.dmi ? { v: IDX.dmi, d: null } : null;
    if (pid === 'mjo') return IDX.mjo && IDX.mjo[C.MJO_VAR] ? { v: IDX.mjo[C.MJO_VAR], d: IDX.days && IDX.days.mjo } : null;
    return IDX[pid] ? { v: IDX[pid], d: IDX.days && IDX.days[pid] } : null;
  }
  const prevZ = (y, mo, k) => { const z = y * 12 + mo - 1 - k; return { y: Math.floor(z / 12), mo: z % 12 + 1 }; };
  function nilai1(S, y, mo, boleh) {
    const k = y + '-' + mo, v = S.v[k];
    if (v === undefined || v === null || !isFinite(v)) return null;
    const n = S.d ? S.d[k] : null;
    if (n != null) {
      const now_ = boleh && y * 12 + mo - 1 === NOWZ;
      if (n < (now_ ? C.MIN_DAYS_NOW : C.MIN_DAYS)) return null;
    }
    return +v;
  }
  /* IOD: bulan setelah data terakhir diisi persistensi yang meluruh cepat (IOD tidak persisten) */
  function iodMap() {
    if (!IDX || !IDX.dmi) return {};
    if (iodC && iodC.src === IDX) return iodC.map;
    const map = Object.assign({}, IDX.dmi);
    let L = -1;
    Object.keys(map).forEach(k => { const p = k.split('-').map(Number); L = Math.max(L, p[0] * 12 + p[1] - 1); });
    if (L >= 0) {
      const v = map[Math.floor(L / 12) + '-' + (L % 12 + 1)];
      for (let h = 1; h <= 40; h++) {
        const z = L + h, k = Math.floor(z / 12) + '-' + (z % 12 + 1);
        if (map[k] === undefined) map[k] = v * Math.exp(-h / C.TAU_IOD);
      }
    }
    iodC = { src: IDX, map: map };
    return map;
  }
  /* prediktor mentah untuk data LATIH (hanya nilai teramati) */
  function rawTrain(p, y, mo) {
    if (p.id === 'enso') return lagE(y, mo, ENS.obs);
    if (p.id === 'iod') return IDX && IDX.dmi ? lagE(y, mo, IDX.dmi) : null;
    const S = serie(p.id); if (!S) return null;
    const q = prevZ(y, mo, 1);
    return nilai1(S, q.y, q.mo, false);
  }

  /* ================= MODEL: IRLS + RIDGE ================= */
  function solve(A, r) {
    const n = r.length, M = A.map((row, i) => row.concat([r[i]]));
    for (let c = 0; c < n; c++) {
      let p = c; for (let i = c + 1; i < n; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i;
      if (Math.abs(M[p][c]) < 1e-12) return null;
      const t = M[c]; M[c] = M[p]; M[p] = t;
      for (let i = c + 1; i < n; i++) { const f = M[i][c] / M[c][c]; for (let k = c; k <= n; k++) M[i][k] -= f * M[c][k]; }
    }
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let k = i + 1; k < n; k++) s -= M[i][k] * x[k]; x[i] = s / M[i][i]; }
    return x;
  }
  function irls(X, y, lam, init, maxIt) {
    const n = X.length, p = X[0].length, a0 = Math.log(Math.max(mean(y), 1e-6));
    let b = init ? init.slice(0, p) : new Array(p).fill(0);
    if (!init || b.length < p) { b = new Array(p).fill(0); b[0] = a0; }
    for (let it = 0; it < maxIt; it++) {
      const w = new Array(n), z = new Array(n); let sw = 0;
      for (let i = 0; i < n; i++) {
        let eta = 0; for (let j = 0; j < p; j++) eta += b[j] * X[i][j];
        eta = clamp(eta, a0 - 3, a0 + 3);
        const mu = Math.exp(eta); w[i] = mu; z[i] = eta + (y[i] - mu) / mu; sw += mu;
      }
      sw /= n;
      const A = Array.from({ length: p }, () => new Array(p).fill(0)), r = new Array(p).fill(0);
      for (let i = 0; i < n; i++) {
        const wi = w[i] / sw;
        for (let j = 0; j < p; j++) { const xj = X[i][j] * wi; r[j] += xj * z[i]; for (let k = 0; k <= j; k++) A[j][k] += xj * X[i][k]; }
      }
      for (let j = 0; j < p; j++) { for (let k = j + 1; k < p; k++) A[j][k] = A[k][j]; if (j > 0) A[j][j] += lam; }
      const nb = solve(A, r); if (!nb) break;
      let d = 0; for (let j = 0; j < p; j++) d = Math.max(d, Math.abs(nb[j] - b[j]));
      b = nb; if (d < 1e-7) break;
    }
    return b;
  }
  const zOf = (x, mu, sd) => x === null || x === undefined ? 0 : clamp((x - mu) / sd, -C.ZMAX, C.ZMAX);
  function prep(rows, S, raw) {
    const mu = [], sd = [];
    S.forEach(j => {
      const v = rows.map(i => raw[i][j]).filter(x => x !== null);
      const m = v.length ? mean(v) : 0, s = v.length > 1 ? Math.sqrt(sum(v.map(x => (x - m) ** 2)) / (v.length - 1)) : 1;
      mu.push(m); sd.push(s > 1e-9 ? s : 1);
    });
    return { mu: mu, sd: sd, X: rows.map(i => [1].concat(S.map((j, q) => zOf(raw[i][j], mu[q], sd[q])))) };
  }
  function looModel(S, raw, y, all) {
    const n = y.length, bf = irls(prep(all, S, raw).X, y, C.LAMBDA, null, 40), preds = new Array(n);
    for (let i = 0; i < n; i++) {
      const tr = all.filter(k => k !== i), P = prep(tr, S, raw), yt = tr.map(k => y[k]), m = mean(yt);
      const b = irls(P.X, yt, C.LAMBDA, bf, 12);
      const x = [1].concat(S.map((j, q) => zOf(raw[i][j], P.mu[q], P.sd[q])));
      let eta = 0; for (let k = 0; k < x.length; k++) eta += x[k] * b[k];
      preds[i] = m * clamp(Math.exp(clamp(eta, Math.log(m) - 3, Math.log(m) + 3)) / m, F_LO, F_HI);
    }
    return { mse: mean(y.map((v, i) => (v - preds[i]) ** 2)), preds: preds };
  }

  /* ================= FIT: inti (dipakai model bulanan dan 3 bulanan) ================= */
  const label = S => 'Baseline' + S.map(j => ' + ' + PRED[j].nm).join('');
  /* yrs[i], y[i] = tahun & total (mm); raw[i][j] = prediktor ke-j (null bila tidak ada); allow[j] = boleh dipakai */
  function fitCore(yrs, y, raw, allow, cvMin) {
    const n = y.length;
    if (n < 3) return null;
    const ybar = mean(y);
    if (!(ybar > 0)) return null;
    const sy = sum(y), loC = y.map(v => (sy - v) / (n - 1)), mseC = mean(y.map((v, i) => (v - loC[i]) ** 2));
    const cvOf = m => clamp(Math.sqrt(m) / ybar, cvMin, 1.5);
    const F = { n: n, mean: ybar, ok: false, t: 0, a: Math.log(ybar), b: 0, cv: cvOf(mseC), cv0: cvOf(mseC), msess: 0, model: 'Baseline',
      set: [], idxS: [], mu: [], sd: [], beta: [], abl: [], usable: [], obs: {}, loo: {}, looC: {} };
    yrs.forEach((yy, i) => { F.obs[yy] = y[i]; F.looC[yy] = loC[i]; F.loo[yy] = loC[i]; });
    if (n < MIN_N) return F;

    const usable = [];
    PRED.forEach((p, j) => { if (allow[j] && raw.filter(r => r[j] !== null).length >= Math.max(MIN_N, C.MIN_COVER * n)) usable.push(j); });
    F.usable = usable.map(j => PRED[j].id);
    if (!usable.length) return F;

    const all = y.map((_, i) => i), chain = [[]];
    usable.forEach((j, k) => chain.push(usable.slice(0, k + 1)));
    const res = chain.map((S, k) => k ? looModel(S, raw, y, all) : null);
    const mses = chain.map((S, k) => k ? res[k].mse : mseC);
    const sq = chain.map((S, k) => k ? y.map((v, i) => (v - res[k].preds[i]) ** 2) : y.map((v, i) => (v - loC[i]) ** 2));
    const lebihBaik = (a, b) => {   // b mengalahkan a? (rata-rata galat turun >= MARGIN dan selisihnya signifikan secara berpasangan)
      const d = a.map((v, i) => v - b[i]), m = mean(d), s = Math.sqrt(sum(d.map(x => (x - m) ** 2)) / (n - 1));
      return mean(b) < mean(a) * (1 - C.MARGIN) && m > 0 && m / Math.max(1e-12, s / Math.sqrt(n)) >= C.ZSEL;
    };
    let best = 0;
    for (let k = 1; k < chain.length; k++) if (lebihBaik(sq[best], sq[k])) best = k;
    F.abl = chain.map((S, k) => ({ set: S.map(j => PRED[j].id), msess: mseC > 0 ? 1 - mses[k] / mseC : 0, sel: k === best }));
    if (best > 0) {
      const S = chain[best], P = prep(all, S, raw), b = irls(P.X, y, C.LAMBDA, null, 40);
      F.ok = true; F.a = b[0]; F.beta = b.slice(1); F.mu = P.mu; F.sd = P.sd; F.idxS = S;
      F.set = S.map(j => PRED[j].id); F.model = label(S);
      F.msess = mseC > 0 ? 1 - mses[best] / mseC : 0; F.t = F.msess; F.cv = cvOf(mses[best]);
      yrs.forEach((yy, i) => { F.loo[yy] = res[best].preds[i]; });
    }
    return F;
  }
  /* model BULANAN: prediktor lag 1-3 bulan (ENSO/IOD) atau lag 1 bulan (MJO/Kelvin/Rossby) */
  function fit2(id, mo) {
    const cache = A[id].fit;
    if (cache[mo] !== undefined) return cache[mo];
    const R = A[id].rain.filter(t => t.mo === mo && okPos(id, t));
    const raw = R.map(t => PRED.map(p => rawTrain(p, t.y, t.mo)));
    return (cache[mo] = fitCore(R.map(t => t.y), R.map(t => t.mm), raw, PRED.map(() => true), .2));
  }
  fit = function (id, mo) { return fit2(id, mo); };   // satu cache untuk semua pemanggil

  /* ================= PREDIKSI ================= */
  function zTarget(f, m, lead, e, map, imap) {
    const att = 1 - ATT * Math.min(1, Math.max(0, lead - 1) / 11);
    return f.idxS.map((pj, q) => {
      const id = PRED[pj].id; let x = null;
      if (id === 'enso') { x = lagE(m.y, m.mo, map); if (x === null) x = e; }
      else if (id === 'iod') x = lagE(m.y, m.mo, imap);
      else { const S = serie(id); if (S) { const p = prevZ(m.y, m.mo, 1); x = nilai1(S, p.y, p.mo, true); } }
      if (x === null || x === undefined) return 0;
      const z = clamp((x - f.mu[q]) / f.sd[q], -C.ZMAX, C.ZMAX);
      return id === 'enso' || id === 'iod' ? z * att : z;
    });
  }
  function peluang(Nm, cv0, ds) {
    let pb = 0, pa = 0;
    if (ds.a !== null) {
      const a0 = 1 / (cv0 * cv0), th0 = Nm / a0, lo = gq(1 / 3, a0, th0), hi = gq(2 / 3, a0, th0);
      pb = clamp(gp(ds.a, lo / ds.th), 0, 1); pa = clamp(1 - gp(ds.a, hi / ds.th), 0, 1);
    }
    const pn = Math.max(0, 1 - pb - pa);
    return { pb: pb, pn: pn, pa: pa, cat: pb >= .4 && pb >= pa ? 'bn' : pa >= .4 ? 'an' : 'n' };
  }
  compute = function (id, list, ev) {
    const map = lagMap(), imap = iodMap();
    return list.map((m, i) => {
      const f = fit2(id, m.mo), e = ev[i];
      if (!f) return { m: m, e: e, none: true };
      const lead = Math.max(0, m.y * 12 + m.mo - 1 - NOWZ), Nm = f.mean;
      const capped = f.ok && lead > C.MONTHLY_MAX_LEAD, pakai = f.ok && !capped;   // lead panjang: prediktor lag 1-3 bln belum teramati
      let F = 1, z = [];
      if (pakai) {
        z = zTarget(f, m, lead, e, map, imap);
        let eta = f.a; f.beta.forEach((b, q) => { eta += b * z[q]; });
        F = clamp(Math.exp(eta) / Nm, F_LO, F_HI);
      }
      const mu = Nm * F, cv0 = f.cv0, cvF = pakai ? Math.min(cv0, Math.sqrt(f.cv ** 2 + (.04 * lead) ** 2)) : cv0;
      const ds = dist(mu, cvF), pl = peluang(Nm, cv0, ds);
      return { m: m, e: e, F: F, n: f.n, t: f.t, fitOK: pakai, capped: capped, Nm: Nm, mean: mu, p10: ds.p10, p90: ds.p90, pb: pl.pb, pn: pl.pn, pa: pl.pa, cat: pl.cat, z: z };
    });
  };

  /* Kontribusi berurutan (mm): Baseline + dENSO + dIOD + dMJO + dKelvin + dRossby = Prediksi.
     Urutan memengaruhi pembagian karena model berbentuk perkalian; hasil dinormalkan agar jumlahnya pas. */
  function kontribusi(x, f) {
    if (!f || !f.ok || !x.z || x.capped || !x.z.length) return null;
    const out = {}; let acc = 0, gPrev = 1, tot = 0;
    f.idxS.forEach((pj, q) => { acc += f.beta[q] * x.z[q]; const g = Math.exp(acc); out[pj] = x.Nm * (g - gPrev); gPrev = g; tot += out[pj]; });
    const want = x.mean - x.Nm;
    if (tot * want > 0) { const s = clamp(want / tot, .5, 2); Object.keys(out).forEach(k => { out[k] *= s; }); }
    return out;
  }

  /* ================= 3 BULANAN: MODEL PER JARAK PRAKIRAAN =================
     Target: total hujan 3 bulan berurutan. Prediktor ENSO/IOD = rata-rata 3 bulan yang berakhir pada
     bulan terakhir yang SUDAH TERAMATI, dengan jeda (lag) yang sama persis pada data latih. Jadi El Niño
     tidak perlu diekstrapolasi ke masa depan; model belajar langsung dari sejarah "kondisi sekarang ->
     hujan N bulan kemudian". MJO/Kelvin/Rossby hanya ikut bila nilainya sudah ada untuk bulan tepat
     sebelum periode (praktis: hanya periode yang dimulai bulan depan). */
  const ABR = BLN.map(b => b.slice(0, 3));
  const WAVES = ['mjo', 'kelvin', 'rossby'];
  const zKey = z => Math.floor(z / 12) + '-' + (z % 12 + 1);
  const zMo = z => ({ y: Math.floor(z / 12), mo: z % 12 + 1 });
  function lastZ(map) {
    let L = -1;
    Object.keys(map || {}).forEach(k => { if (!Number.isFinite(map[k])) return; const p = k.split('-').map(Number), z = p[0] * 12 + p[1] - 1; if (z > L) L = z; });
    return L;
  }
  function mean3(map, zEnd) {
    const v = [0, 1, 2].map(k => map[zKey(zEnd - k)]).filter(Number.isFinite);
    return v.length >= 2 ? mean(v) : null;
  }
  function ridx(id) {
    const c = A[id].fit;
    if (!c._r) { const m = {}; A[id].rain.forEach(t => { if (okPos(id, t)) m[t.y * 12 + t.mo - 1] = t.mm; }); c._r = m; }
    return c._r;
  }
  const WIN = Array.from({ length: 10 }, (_, i) => mk(NOWZ + 1 + i));   // periode 3 bulan yang dimulai Des..Sep (menutup 12 bulan ke depan)
  function lab(s) { const e = zMo(s.y * 12 + s.mo + 1); return ABR[s.mo - 1] + ' ' + s.y + '–' + ABR[e.mo - 1] + ' ' + e.y; }

  function musim(id, s) {
    const z0 = s.y * 12 + s.mo - 1, lead = z0 - NOWZ;
    const eL = lastZ(ENS.obs), iL = IDX && IDX.dmi ? lastZ(IDX.dmi) : -1;
    const lagE = eL >= 0 ? z0 - eL : -1, lagI = iL >= 0 ? z0 - iL : -1;
    const okE = lagE >= 1 && lagE <= C.SEAS_MAX_LAG, okI = lagI >= 1 && lagI <= C.SEAS_MAX_LAG;
    const pv = zMo(z0 - 1);
    const wv = WAVES.map(w => { const S = serie(w); return S ? nilai1(S, pv.y, pv.mo, true) : null; });
    const ck = 's|' + s.mo + '|' + (okE ? lagE : 'x') + '|' + (okI ? lagI : 'x') + '|' + wv.map(v => v === null ? 0 : 1).join('');
    const cache = A[id].fit;
    let f = cache[ck];
    if (f === undefined) {
      const r = ridx(id), yrs = [], tot = [], raw = [];
      Object.keys(r).map(Number).filter(z => z % 12 + 1 === s.mo && r[z + 1] !== undefined && r[z + 2] !== undefined).sort((a, b) => a - b).forEach(z => {
        yrs.push(Math.floor(z / 12)); tot.push(r[z] + r[z + 1] + r[z + 2]);
        const p1 = zMo(z - 1);
        raw.push([okE ? mean3(ENS.obs, z - lagE) : null, okI ? mean3(IDX.dmi, z - lagI) : null].concat(WAVES.map((w, q) => {
          if (wv[q] === null) return null;
          return nilai1(serie(w), p1.y, p1.mo, false);
        })));
      });
      f = cache[ck] = fitCore(yrs, tot, raw, [okE, okI, wv[0] !== null, wv[1] !== null, wv[2] !== null], .1);
    }
    if (!f) {
      const r = ridx(id);
      const nw = Object.keys(r).map(Number).filter(z => z % 12 + 1 === s.mo && r[z + 1] !== undefined && r[z + 2] !== undefined).length;
      return { s: s, lead: lead, none: true, nwin: nw };
    }
    const xs = [okE ? mean3(ENS.obs, z0 - lagE) : null, okI ? mean3(IDX.dmi, z0 - lagI) : null].concat(wv);
    const Nm = f.mean;
    let F = 1, z = [];
    if (f.ok) {
      z = f.idxS.map((j, q) => xs[j] === null ? 0 : clamp((xs[j] - f.mu[q]) / f.sd[q], -C.ZMAX, C.ZMAX));
      let eta = f.a; f.beta.forEach((b, q) => { eta += b * z[q]; });
      F = clamp(Math.exp(eta) / Nm, F_LO, F_HI);
    }
    const mu = Nm * F, cv0 = f.cv0, ds = dist(mu, f.ok ? f.cv : cv0), pl = peluang(Nm, cv0, ds);
    return { s: s, lead: lead, f: f, Nm: Nm, mean: mu, p10: ds.p10, p90: ds.p90, pb: pl.pb, pn: pl.pn, pa: pl.pa, cat: pl.cat, z: z,
      lagE: okE ? lagE : null, lagI: okI ? lagI : null, eL: eL, iL: iL };
  }

  /* ================= TAHUNAN ================= */
  function annual(id) {
    const c = A[id].fit;
    if (c._ann !== undefined) return c._ann;
    const fs = []; for (let mo = 1; mo <= 12; mo++) fs.push(fit2(id, mo));
    if (fs.some(f => !f)) return (c._ann = null);
    const yrs = Object.keys(fs[0].obs).filter(yy => fs.every(f => f.obs[yy] !== undefined));
    if (yrs.length < MIN_N) return (c._ann = null);
    let sO = 0, em = 0, ec = 0;
    yrs.forEach(yy => {
      const o = sum(fs.map(f => f.obs[yy])), p = sum(fs.map(f => f.loo[yy])), q = sum(fs.map(f => f.looC[yy]));
      sO += o; em += (o - p) ** 2; ec += (o - q) ** 2;
    });
    const n = yrs.length;
    return (c._ann = { n: n, mean: sO / n, rmse: Math.sqrt(em / n), rmseC: Math.sqrt(ec / n), msess: ec > 0 ? 1 - em / ec : 0 });
  }
  const interval = (tp, sd) => {
    if (!(tp > 0) || sd == null) return { lo: null, hi: null };
    const d = dist(tp, clamp(sd / tp, .05, 1.5));
    return { lo: d.p10, hi: d.p90 };
  };
  tahunan = function (id) {
    const ms = compute(id, CAL, ensoAutoFor(CAL).map(r => r.v)), rows = rowsOf(id, REPORT_YEAR);
    let tb = 0, tp = 0, nO = 0, n = 0, rem = 0;
    ms.forEach((c, i) => {
      if (c.none) return; n++; tb += c.Nm;
      const r = rows[i];
      if (r && r.lengkap && valid(r)) { tp += r.mm; nO++; }
      else if (r && valid(r) && r.hari > 0) { const fr = Math.max(0, r.hari_bulan - r.hari) / r.hari_bulan; tp += r.mm + c.mean * fr; rem += c.Nm * fr; }
      else { tp += c.mean; rem += c.Nm; }
    });
    const an = annual(id); let lo = null, hi = null;
    if (an && tb > 0) {
      if (rem <= 0) { lo = hi = tp; }
      else { const iv = interval(tp, an.rmse * Math.sqrt(Math.min(1, rem / tb))); lo = iv.lo; hi = iv.hi; }
    }
    return { tb: tb, tp: tp, nO: nO, n: n, lo: lo, hi: hi, sk: an ? an.msess : null, rmse: an ? an.rmse : null };
  };

  /* ================= TAMPILAN ================= */
  function statusIdx() {
    const o = ['ENSO ✓ (' + (ENS.kind || 'Niño-3.4') + ')'];
    PRED.slice(1).forEach(p => {
      const S = serie(p.id);
      if (!S) { o.push(p.nm + ' ✗ (belum ada di monthly.json)'); return; }
      const ks = Object.keys(S.v).map(k => k.split('-').map(Number)).sort((a, b) => a[0] * 12 + a[1] - (b[0] * 12 + b[1]));
      const l = ks[ks.length - 1]; o.push(p.nm + ' ✓ s.d. ' + BLN[l[1] - 1].slice(0, 3) + ' ' + l[0]);
    });
    return o.join(' · ');
  }
  function htmlKotak() {
    const f12 = PM.map(x => x.none ? null : fit2(kab, x.m.mo));
    const tot = { b: 0, d: PRED.map(() => 0), p: 0 };
    const rows = PM.map((x, i) => {
      if (x.none) return '<tr><td>' + bl(x.m) + '</td><td colspan="9" class="na">data historis belum cukup</td></tr>';
      const f = f12[i], dl = kontribusi(x, f);
      tot.b += x.Nm; tot.p += x.mean;
      const cells = PRED.map((p, j) => {
        if (dl && dl[j] !== undefined) { tot.d[j] += dl[j]; return '<td style="color:' + (dl[j] < 0 ? 'var(--c1)' : 'var(--c2)') + '">' + sgm(dl[j]) + '</td>'; }
        const alasan = x.capped ? 'lead terlalu panjang untuk model bulanan' : f.usable.indexOf(p.id) < 0 ? 'data belum tersedia' : 'tidak menurunkan galat LOYO';
        return '<td class="na" title="' + p.nm + ': ' + alasan + '">–</td>';
      }).join('');
      return '<tr><td>' + bl(x.m) + '</td><td>' + fmt(x.Nm, 0) + '</td>' + cells + '<td><b>' + fmt(x.mean, 0) + '</b></td><td class="na">' +
        (x.capped ? 'Baseline (lead > ' + C.MONTHLY_MAX_LEAD + ' bln)' : f.ok ? f.model : 'Baseline') + '</td><td>' + (f.ok && !x.capped ? sg2(f.msess) : '–') + '</td></tr>';
    }).join('');
    const foot = '<tr><td>Total 12 bln</td><td>' + fmt(tot.b, 0) + '</td>' + tot.d.map(v => '<td>' + (v ? sgm(v) : '–') + '</td>').join('') +
      '<td>' + fmt(tot.p, 0) + '</td><td colspan="2"></td></tr>';

    const ev = ensoAutoFor(MONTHS).map(r => r.v);
    let nm = 0, pakai = PRED.map(() => 0);
    KAB.forEach(k => { for (let mo = 1; mo <= 12; mo++) { const f = fit2(k.id, mo); if (!f) continue; nm++; f.idxS.forEach(j => { pakai[j]++; }); } });
    const ann = KAB.map(k => {
      const ms = compute(k.id, MONTHS, ev).filter(x => !x.none), ty = tahunan(k.id), an = annual(k.id);
      if (!ms.length) return '<tr><td>' + k.nm + '</td><td colspan="8" class="na">data belum cukup</td></tr>';
      const tb12 = sum(ms.map(x => x.Nm)), tp12 = sum(ms.map(x => x.mean)), iv = interval(tp12, an ? an.rmse : null);
      const bm = ms.filter(x => x.fitOK).length;
      return '<tr><td><b>' + k.nm + '</b></td><td>' + fmt(ty.tb, 0) + '</td><td><b>' + fmt(ty.tp, 0) + '</b></td><td>' +
        (ty.lo != null ? fmt(ty.lo, 0) + '–' + fmt(ty.hi, 0) : '–') + '</td><td class="pc">' + (ty.tb ? fmt(ty.tp / ty.tb * 100, 0) + '%' : '–') + '</td><td><b>' +
        fmt(tp12, 0) + '</b></td><td>' + (iv.lo != null ? fmt(iv.lo, 0) + '–' + fmt(iv.hi, 0) : '–') + '</td><td>' + (an ? sg2(an.msess) : '–') +
        '</td><td>' + bm + '/' + ms.length + '</td></tr>';
    }).join('');

    return '<h2>Prediksi bulanan = Baseline + ENSO + IOD + MJO + Kelvin + Rossby · ' + nmKab() + '</h2>' +
      '<p class="note" style="margin:0 0 8px">Prediktor: ' + statusIdx() + '</p>' +
      '<div class="tw"><table id="tbl-pred"><thead><tr><th>Bulan</th><th>Baseline</th>' + PRED.map(p => '<th>Δ ' + p.nm + '</th>').join('') +
      '<th>Prediksi</th><th>Model terpilih</th><th>Skill (MSESS)</th></tr></thead><tbody>' + rows + '</tbody><tfoot>' + foot + '</tfoot></table></div>' +
      '<p class="note">Satuan mm. Δ = kontribusi berurutan (ENSO → IOD → MJO → Kelvin → Rossby), dinormalkan sehingga Baseline + ΣΔ = Prediksi. “–” = prediktor tidak dipakai di bulan itu. ' +
      'Skill = 1 − MSE/MSE-klimatologi pada validasi leave-one-year-out; model hanya dipakai bila menurunkan galat ≥ ' + Math.round(C.MARGIN * 100) + '%. ' +
      'Bulan dengan lead > ' + C.MONTHLY_MAX_LEAD + ' bln otomatis memakai Baseline (lihat tabel 3 bulanan di atas). Model tervalidasi dipakai pada ' + PRED.map((p, j) => p.nm + ' ' + pakai[j]).join(', ') + ' dari ' + nm + ' kombinasi kabupaten-bulan. ' +
      'MJO/Kelvin/Rossby memakai lag 1 bulan, sehingga hanya memengaruhi bulan terdekat; bulan lebih jauh dianggap netral.</p>' +
      '<h2 style="margin-top:18px">Prediksi tahunan semua kabupaten</h2>' +
      '<div class="tw"><table id="tbl-ann"><thead><tr><th>Kabupaten</th><th>Normal ' + REPORT_YEAR + '</th><th>Prediksi ' + REPORT_YEAR + '</th><th>P10–P90</th><th>% normal</th>' +
      '<th>Prediksi 12 bln ke depan</th><th>P10–P90</th><th>Skill tahunan</th><th>Bulan bermodel</th></tr></thead><tbody>' + ann + '</tbody></table></div>' +
      '<p class="note">Prediksi ' + REPORT_YEAR + ' = curah hujan teramati + prediksi sisa bulan/hari. Rentang P10–P90 dari galat tahunan LOYO (jumlah prediksi 12 bulan vs total aktual); ' +
      'skill dihitung dengan prediktor yang sudah teramati, sehingga untuk bulan yang jauh ke depan keterampilan sesungguhnya lebih rendah. Pemilihan model dan penilaiannya memakai data yang sama, jadi anggap skill sebagai batas atas.</p>';
  }

  function htmlMusim() {
    const bln = z => z < 0 ? '–' : ABR[z % 12] + ' ' + Math.floor(z / 12);
    const eL = lastZ(ENS.obs), iL = IDX && IDX.dmi ? lastZ(IDX.dmi) : -1;
    const rows = WIN.map(s => {
      const x = musim(kab, s);
      if (x.none) return '<tr><td>' + lab(s) + '</td><td colspan="14" class="na">data historis belum cukup (hanya ' + x.nwin + ' tahun dengan 3 bulan lengkap berturut-turut)</td></tr>';
      const f = x.f, dl = kontribusi(x, f);
      const cells = PRED.map((p, j) => {
        if (dl && dl[j] !== undefined) return '<td style="color:' + (dl[j] < 0 ? 'var(--c1)' : 'var(--c2)') + '">' + sgm(dl[j]) + '</td>';
        const alasan = f.usable.indexOf(p.id) < 0 ? (j < 2 ? 'data belum tersedia' : 'belum teramati untuk bulan sebelum periode ini') : 'tidak menurunkan galat LOYO';
        return '<td class="na" title="' + p.nm + ': ' + alasan + '">–</td>';
      }).join('');
      const jeda = 'ENSO ' + (x.lagE == null ? '–' : x.lagE) + ' · IOD ' + (x.lagI == null ? '–' : x.lagI) + ' bln';
      return '<tr><td><b>' + lab(s) + '</b></td><td>' + x.lead + '</td><td>' + fmt(x.Nm, 0) + '</td>' + cells + '<td><b>' + fmt(x.mean, 0) + '</b></td><td>' +
        fmt(x.p10, 0) + '–' + fmt(x.p90, 0) + '</td><td class="pc">' + fmt(x.mean / x.Nm * 100, 0) + '%</td><td><div class="pr"><div class="pb"><i style="width:' + x.pb * 100 +
        '%;background:var(--c1)"></i><i style="width:' + x.pn * 100 + '%;background:var(--mut)"></i><i style="width:' + x.pa * 100 + '%;background:var(--c2)"></i></div><span>' +
        Math.round(x.pb * 100) + '/' + Math.round(x.pn * 100) + '/' + Math.round(x.pa * 100) + '%</span></div></td><td><span class="cat ' + x.cat + '">' + (f.ok ? CN[x.cat] : 'Klimatologi') + '</span></td><td class="na">' +
        (f.ok ? f.model : 'Baseline') + '<span class="ps">jeda ' + jeda + ' · n ' + f.n + '</span></td><td>' + (f.ok ? sg2(f.msess) : '–') + '</td></tr>';
    }).join('');

    const STD = WIN.filter(s => [12, 3, 6, 9].indexOf(s.mo) > -1);
    let nOk = 0, nAll = 0;
    const kurang = {};
    const all = KAB.map(k => {
      const tds = STD.map(s => {
        const x = musim(k.id, s);
        if (x.none) { kurang[k.nm] = 1; return '<td class="na" title="hanya ' + x.nwin + ' tahun dengan 3 bulan lengkap berturut-turut">–<span class="ps">n=' + x.nwin + '</span></td>'; }
        nAll++; if (x.f.ok) nOk++;
        return '<td><b>' + fmt(x.mean, 0) + '</b> mm · ' + fmt(x.mean / x.Nm * 100, 0) + '%<span class="ps"><span class="cat ' + x.cat + '">' + (x.f.ok ? CN[x.cat] : 'Klimatologi') + '</span> ' +
          (x.f.ok ? 'skill ' + sg2(x.f.msess) : 'Baseline') + '</span></td>';
      }).join('');
      return '<tr><td><b>' + k.nm + '</b></td>' + tds + '</tr>';
    }).join('');

    return '<h2>Prediksi 3 bulanan untuk 6–12 bulan ke depan · ' + nmKab() + '</h2>' +
      '<p class="note" style="margin:0 0 8px">Model dilatih per jarak prakiraan. ENSO terakhir teramati: <b>' + bln(eL) + '</b> · IOD terakhir teramati: <b>' + bln(iL) +
      '</b>. Prediktornya adalah rata-rata 3 bulan terakhir itu, dengan jeda yang sama persis pada data latih, jadi tidak ada perkiraan ENSO ke depan.</p>' +
      '<div class="tw"><table id="tbl-seas"><thead><tr><th>Periode</th><th>Lead</th><th>Baseline</th>' + PRED.map(p => '<th>Δ ' + p.nm + '</th>').join('') +
      '<th>Prediksi</th><th>P10–P90</th><th>% normal</th><th>Peluang BN / N / AN</th><th>Kategori</th><th>Model terpilih</th><th>Skill (MSESS)</th></tr></thead><tbody>' + rows + '</tbody></table></div>' +
      '<p class="note">Satuan mm; Lead = bulan dari sekarang ke awal periode. Δ = kontribusi berurutan (ENSO → IOD → MJO → Kelvin → Rossby) sehingga Baseline + ΣΔ = Prediksi. ' +
      'Model hanya dipakai bila galat leave-one-year-out turun ≥ ' + Math.round(C.MARGIN * 100) + '% dan signifikan; selain itu hasilnya Baseline. ' +
      'MJO/Kelvin/Rossby praktis hanya bisa ikut pada periode yang dimulai bulan depan, karena pengaruhnya hilang dalam hitungan minggu. ' +
      'Periode yang saling tumpang-tindih tidak independen.</p>' +
      '<h2 style="margin-top:18px">Ringkasan musiman semua kabupaten</h2>' +
      '<div class="tw"><table id="tbl-seas-all"><thead><tr><th>Kabupaten</th>' + STD.map(s => '<th>' + lab(s) + '</th>').join('') + '</tr></thead><tbody>' + all + '</tbody></table></div>' +
      '<p class="note">Model prediktor terpilih pada ' + nOk + ' dari ' + nAll + ' kombinasi kabupaten-musim. Pada uji tanpa sinyal, sekitar 3–5% kombinasi tetap lolos karena kebetulan, ' +
      'jadi angka kecil di sini wajar dan sebagian besar bukan hubungan nyata. <b>Baseline/“Klimatologi” bukan prakiraan “normal”</b>: artinya tidak ada prediktor yang terbukti memperbaiki prakiraan, sehingga yang ditampilkan hanya rata-rata historis dan rentang ketidakpastiannya.' +
      (Object.keys(kurang).length ? ' Data kurang (tanda “–”, kurang dari 3 tahun dengan tiga bulan lengkap berturut-turut) pada sebagian/seluruh musim: ' + Object.keys(kurang).join(', ') + '.' : '') + '</p>';
  }

  /* ================= HOMOGENITAS DATA BBWS ================= */
  function posStat(id) {
    const ys = A[id].years, out = [];
    Object.keys(ys).map(Number).sort((a, b) => a - b).forEach(y => {
      const v = ys[y].filter(r => r && r.pos_rata != null && r.pos_rata > 0);
      if (v.length) out.push({ y: y, pos: mean(v.map(r => r.pos_rata)), n: v.length, lengkap: ys[y].filter(r => r && r.lengkap).length });
    });
    return out;
  }
  function infoK(id) {
    const ys = A[id].years, st = posStat(id);
    if (!st.length) return null;
    let z0 = Infinity, z1 = -Infinity, lengkap = 0;
    Object.keys(ys).forEach(y => ys[y].forEach((r, i) => { if (!r) return; const z = +y * 12 + i; z0 = Math.min(z0, z); z1 = Math.max(z1, z); if (r.lengkap) lengkap++; }));
    return { st: st, areal: isArealK(id), z0: z0, z1: z1, lengkap: lengkap, span: z1 - z0 + 1, pakai: A[id].rain.filter(t => okPos(id, t)).length, total: A[id].rain.length };
  }
  function htmlHomogen() {
    const bln = z => ABR[z % 12] + ' ' + Math.floor(z / 12);
    const info = KAB.map(k => ({ k: k, i: infoK(k.id) }));
    const nAreal = info.filter(o => o.i && o.i.areal).length, nSt = info.filter(o => o.i && !o.i.areal).length;
    const rows = info.map(o => {
      const i = o.i, k = o.k;
      if (!i) return '<tr><td>' + k.nm + '</td><td colspan="7" class="na">belum ada data</td></tr>';
      const pct = i.lengkap / i.span * 100, pf = mean(i.st.slice(0, 3).map(x => x.pos)), pl = mean(i.st.slice(-3).map(x => x.pos));
      return '<tr><td><b>' + k.nm + '</b></td><td>' + (i.areal ? 'Areal (pos = 1)' : 'Stasiun') + '</td><td>' + bln(i.z0) + '</td><td>' + bln(i.z1) + '</td><td>' + i.lengkap + ' / ' + i.span +
        '</td><td' + (pct < 80 ? ' style="color:var(--warm);font-weight:700"' : '') + '>' + fmt(pct, 0) + '%</td><td>' + (i.areal ? '–' : fmt(pf, 1) + ' → ' + fmt(pl, 1)) + '</td><td>' + (i.areal ? '–' : fmt(.5 * pl, 1)) + '</td></tr>';
    }).join('');
    let ctrl = '', per = '';
    if (nSt) {
      ctrl = '<p style="margin:10px 0 0"><label>Pos aktif minimum untuk data latih (hanya kabupaten berjenis Stasiun): <input type="number" id="pos-min" min="0" step="0.5" value="' + C.POS_MIN + '" style="max-width:90px"></label> ' +
        '<span class="note">0 = pakai semua data. “Saran” = setengah dari rata-rata pos terkini.</span></p>';
      const ik = infoK(kab);
      if (ik && !ik.areal) per = '<details style="margin-top:10px"><summary>Rincian per tahun · ' + nmKab() + '</summary><div class="tw"><table id="tbl-homog-th"><thead><tr><th>Tahun</th><th>Pos rata-rata</th><th>Bulan lengkap</th><th>Dipakai</th></tr></thead><tbody>' +
        ik.st.map(x => '<tr><td>' + x.y + '</td><td>' + fmt(x.pos, 1) + '</td><td>' + x.lengkap + ' / 12</td><td>' + (C.POS_MIN > 0 && x.pos < C.POS_MIN ? '<span class="na">dikeluarkan</span>' : '✓') + '</td></tr>').join('') + '</tbody></table></div></details>';
    }
    return '<h2>Kualitas dan cakupan data latih</h2>' +
      '<p class="note" style="margin:0 0 8px">Jenis data dikenali dari jumlah pos: <b>Areal</b> = jumlah pos selalu 1 (nilai rata-rata wilayah/grid, mis. dari GEE; tidak ada ketidakhomogenan akibat penambahan pos), ' +
      '<b>Stasiun</b> = rata-rata beberapa pos telemetri. Saat ini: ' + nAreal + ' kabupaten areal, ' + nSt + ' kabupaten stasiun. Kelengkapan = bulan lengkap ÷ rentang bulan sejak data pertama (merah bila &lt; 80%); bulan yang tidak lengkap tidak bisa dipakai sebagai data latih.</p>' +
      '<div class="tw"><table id="tbl-homog"><thead><tr><th>Kabupaten</th><th>Jenis data</th><th>Mulai</th><th>Terakhir</th><th>Bulan lengkap / rentang</th><th>Kelengkapan</th><th>Pos rata-rata awal → kini</th><th>Saran pos minimum</th></tr></thead><tbody>' + rows + '</tbody></table></div>' + ctrl + per;
  }

  function pasca2() {
    if (mode !== 'proy') return;
    document.querySelectorAll('#tbl-proyeksi thead th').forEach(th => { if (th.textContent.trim() === 'n / t') th.textContent = 'n / skill'; });
    const ths = document.querySelectorAll('#tsr thead th'); if (ths[6]) ths[6].textContent = 'Bulan bermodel prediktor';
    document.querySelectorAll('#tpm sup').forEach(s => { s.title = 'Model Baseline: prediktor tidak mengalahkan klimatologi (LOYO), data < ' + MIN_N + ' tahun, atau lead > ' + C.MONTHLY_MAX_LEAD + ' bulan'; });
    const psm = $('psm'); if (psm) psm.textContent += ' Prediktor tambahan: ' + statusIdx() + '.';
    document.querySelectorAll('#tpm tr').forEach((tr, i) => {
      const x = PM[i], c = tr.querySelector('span.cat');
      if (x && !x.none && !x.fitOK && c && x.cat === 'n') c.textContent = 'Klimatologi';
    });
    const cards = document.querySelectorAll('#cp .card');
    if (cards[1]) { const th = tahunan(kab), s = cards[1].querySelector('.s'); if (s && th.lo != null) s.textContent += ' · P10–P90 ' + fmt(th.lo, 0) + '–' + fmt(th.hi, 0) + ' mm'; }
    $('pnt').textContent = '† = Baseline (tanpa koreksi prediktor): tidak ada kombinasi ENSO/IOD/MJO/Kelvin/Rossby yang menurunkan galat leave-one-year-out ≥ ' +
      Math.round(C.MARGIN * 100) + '%, atau data < ' + MIN_N + ' tahun. Kategori memakai tercile klimatologi: Bawah/Atas Normal hanya bila peluangnya ≥ 40%.';
    let bh = $('box-homog');
    if (!bh) { bh = document.createElement('section'); bh.className = 'box'; bh.id = 'box-homog'; $('vp').appendChild(bh); }
    bh.innerHTML = htmlHomogen();
    let bs = $('box-seas');
    if (!bs) { bs = document.createElement('section'); bs.className = 'box'; bs.id = 'box-seas'; $('vp').appendChild(bs); }
    bs.innerHTML = htmlMusim();
    let box = $('box-pred');
    if (!box) { box = document.createElement('section'); box.className = 'box'; box.id = 'box-pred'; $('vp').appendChild(box); }
    box.innerHTML = htmlKotak();
  }
  const _render = render;
  render = function () { _render(); try { pasca2(); } catch (e) { console.warn('prediktor.js:', e); } };

  /* ================= MUAT monthly.json ================= */
  function resetFit() { KAB.forEach(k => { if (A[k.id]) A[k.id].fit = {}; }); }
  const adaIdx = d => d && (d.dmi || d.mjo || d.kelvin || d.rossby);
  async function muatIdx() {
    const c = bacaLS(C.LS); if (adaIdx(c)) { IDX = c; iodC = null; resetFit(); }
    try {
      const j = JSON.parse(await ft(C.URL + '?t=' + Math.floor(Date.now() / 600000), 10000));
      if (adaIdx(j)) { IDX = j; iodC = null; tulisLS(C.LS, j); }
    } catch (e) { console.warn('prediktor.js: monthly.json tidak termuat —', e.message); }
    resetFit();
    if (!$('app').hidden && tahun != null) render();
  }
  document.addEventListener('change', e => {
    if (!e.target || e.target.id !== 'pos-min') return;
    const v = parseFloat(e.target.value);
    C.POS_MIN = isFinite(v) && v > 0 ? v : 0; tulisLS('ch_posmin_v1', C.POS_MIN);
    resetFit(); render();
  });
  window.PREDIKTOR = { htmlHomogen: htmlHomogen, C: C, musim: musim, htmlMusim: htmlMusim, WIN: WIN, nilai1: nilai1, serie: serie, fit2: fit2, annual: annual, kontribusi: kontribusi, setIdx: d => { IDX = d; iodC = null; resetFit(); }, htmlKotak: htmlKotak };
  muatIdx();
  if (typeof RAW !== 'undefined' && RAW && Object.keys(RAW).length && tahun != null) { resetFit(); render(); }
})();
