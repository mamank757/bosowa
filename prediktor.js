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

   Catatan penting: MJO/Kelvin/Rossby memakai lag 1 bulan. Untuk bulan yang prediktor
   lag-1 nya belum teramati, nilainya dianggap netral (tidak berpengaruh). */
(function () {
  'use strict';

  const C = {
    URL: 'https://raw.githubusercontent.com/mamank757/dasarian/refs/heads/main/data/indices/monthly.json',
    LS: 'ch_idx_v1',
    LAMBDA: 8,          // penalti ridge (satuan prediktor terstandar; setara SHRINK pada kode lama)
    MARGIN: 0.03,       // dan MSE LOYO harus turun >= MARGIN (relatif)
    ZSEL: 1.5,          // selisih galat kuadrat per tahun (LOYO) harus >= ZSEL x galat baku (uji berpasangan)
    TAU_IOD: 2,         // peluruhan IOD (bulan) untuk bulan yang belum teramati
    ZMAX: 2.5,          // batas |z| prediktor terstandar
    MJO_VAR: 'mc',      // fraksi hari MJO aktif di fase 4-5 (Maritime Continent)
    MIN_DAYS: 20,       // bulan harian dianggap lengkap bila >= 20 hari
    MIN_DAYS_NOW: 10,   // bulan berjalan boleh dipakai bila >= 10 hari
    MIN_COVER: 0.6      // prediktor dipakai bila tersedia pada >= 60% tahun latih
  };
  const PRED = [
    { id: 'enso', nm: 'ENSO' },
    { id: 'iod', nm: 'IOD' },
    { id: 'mjo', nm: 'MJO' },
    { id: 'kelvin', nm: 'Kelvin' },
    { id: 'rossby', nm: 'Rossby' }
  ];
  let IDX = null, iodC = null;

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

  /* ================= FIT per kabupaten x bulan ================= */
  const label = S => 'Baseline' + S.map(j => ' + ' + PRED[j].nm).join('');
  function fit2(id, mo) {
    const cache = A[id].fit;
    if (cache[mo] !== undefined) return cache[mo];
    const R = A[id].rain.filter(t => t.mo === mo);
    if (R.length < 3) return (cache[mo] = null);
    const y = R.map(t => t.mm), n = y.length, ybar = mean(y);
    if (!(ybar > 0)) return (cache[mo] = null);
    const sy = sum(y), loC = y.map(v => (sy - v) / (n - 1)), mseC = mean(y.map((v, i) => (v - loC[i]) ** 2));
    const cvOf = m => clamp(Math.sqrt(m) / ybar, .2, 1.5);
    const F = { n: n, mean: ybar, ok: false, t: 0, a: Math.log(ybar), b: 0, cv: cvOf(mseC), cv0: cvOf(mseC), msess: 0, model: 'Baseline',
      set: [], idxS: [], mu: [], sd: [], beta: [], abl: [], usable: [], obs: {}, loo: {}, looC: {} };
    R.forEach((t, i) => { F.obs[t.y] = y[i]; F.looC[t.y] = loC[i]; F.loo[t.y] = loC[i]; });
    if (n < MIN_N) return (cache[mo] = F);

    const raw = R.map(t => PRED.map(p => rawTrain(p, t.y, t.mo))), usable = [];
    PRED.forEach((p, j) => { if (raw.filter(r => r[j] !== null).length >= Math.max(MIN_N, C.MIN_COVER * n)) usable.push(j); });
    F.usable = usable.map(j => PRED[j].id);
    if (!usable.length) return (cache[mo] = F);

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
      R.forEach((t, i) => { F.loo[t.y] = res[best].preds[i]; });
    }
    return (cache[mo] = F);
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
  compute = function (id, list, ev) {
    const map = lagMap(), imap = iodMap();
    return list.map((m, i) => {
      const f = fit2(id, m.mo), e = ev[i];
      if (!f) return { m: m, e: e, none: true };
      const lead = Math.max(0, m.y * 12 + m.mo - 1 - NOWZ), Nm = f.mean;
      let F = 1, z = [];
      if (f.ok) {
        z = zTarget(f, m, lead, e, map, imap);
        let eta = f.a; f.beta.forEach((b, q) => { eta += b * z[q]; });
        F = clamp(Math.exp(eta) / Nm, F_LO, F_HI);
      }
      const mu = Nm * F, cv0 = f.cv0, cvF = f.ok ? Math.min(cv0, Math.sqrt(f.cv ** 2 + (.04 * lead) ** 2)) : cv0;
      const ds = dist(mu, cvF);
      let pb = 0, pa = 0;
      if (ds.a !== null) {
        const a0 = 1 / (cv0 * cv0), th0 = Nm / a0, lo = gq(1 / 3, a0, th0), hi = gq(2 / 3, a0, th0);
        pb = clamp(gp(ds.a, lo / ds.th), 0, 1); pa = clamp(1 - gp(ds.a, hi / ds.th), 0, 1);
      }
      const pn = Math.max(0, 1 - pb - pa), cat = pb >= .4 && pb >= pa ? 'bn' : pa >= .4 ? 'an' : 'n';
      return { m: m, e: e, F: F, n: f.n, t: f.t, fitOK: f.ok, Nm: Nm, mean: mu, p10: ds.p10, p90: ds.p90, pb: pb, pn: pn, pa: pa, cat: cat, z: z };
    });
  };

  /* Kontribusi berurutan (mm): Baseline + dENSO + dIOD + dMJO + dKelvin + dRossby = Prediksi.
     Urutan memengaruhi pembagian karena model berbentuk perkalian; hasil dinormalkan agar jumlahnya pas. */
  function kontribusi(x, f) {
    if (!f || !f.ok || !x.z) return null;
    const out = {}; let acc = 0, gPrev = 1, tot = 0;
    f.idxS.forEach((pj, q) => { acc += f.beta[q] * x.z[q]; const g = Math.exp(acc); out[pj] = x.Nm * (g - gPrev); gPrev = g; tot += out[pj]; });
    const want = x.mean - x.Nm;
    if (tot * want > 0) { const s = clamp(want / tot, .5, 2); Object.keys(out).forEach(k => { out[k] *= s; }); }
    return out;
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
        const alasan = f.usable.indexOf(p.id) < 0 ? 'data belum tersedia' : 'tidak menurunkan galat LOYO';
        return '<td class="na" title="' + p.nm + ': ' + alasan + '">–</td>';
      }).join('');
      return '<tr><td>' + bl(x.m) + '</td><td>' + fmt(x.Nm, 0) + '</td>' + cells + '<td><b>' + fmt(x.mean, 0) + '</b></td><td class="na">' +
        (f.ok ? f.model : 'Baseline') + '</td><td>' + (f.ok ? sg2(f.msess) : '–') + '</td></tr>';
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
      'Dipakai pada ' + PRED.map((p, j) => p.nm + ' ' + pakai[j]).join(', ') + ' dari ' + nm + ' kombinasi kabupaten-bulan. ' +
      'MJO/Kelvin/Rossby memakai lag 1 bulan, sehingga hanya memengaruhi bulan terdekat; bulan lebih jauh dianggap netral.</p>' +
      '<h2 style="margin-top:18px">Prediksi tahunan semua kabupaten</h2>' +
      '<div class="tw"><table id="tbl-ann"><thead><tr><th>Kabupaten</th><th>Normal ' + REPORT_YEAR + '</th><th>Prediksi ' + REPORT_YEAR + '</th><th>P10–P90</th><th>% normal</th>' +
      '<th>Prediksi 12 bln ke depan</th><th>P10–P90</th><th>Skill tahunan</th><th>Bulan bermodel</th></tr></thead><tbody>' + ann + '</tbody></table></div>' +
      '<p class="note">Prediksi ' + REPORT_YEAR + ' = curah hujan teramati + prediksi sisa bulan/hari. Rentang P10–P90 dari galat tahunan LOYO (jumlah prediksi 12 bulan vs total aktual); ' +
      'skill dihitung dengan prediktor yang sudah teramati, sehingga untuk bulan yang jauh ke depan keterampilan sesungguhnya lebih rendah. Pemilihan model dan penilaiannya memakai data yang sama, jadi anggap skill sebagai batas atas.</p>';
  }

  function pasca2() {
    if (mode !== 'proy') return;
    document.querySelectorAll('#tbl-proyeksi thead th').forEach(th => { if (th.textContent.trim() === 'n / t') th.textContent = 'n / skill'; });
    const ths = document.querySelectorAll('#tsr thead th'); if (ths[6]) ths[6].textContent = 'Bulan bermodel prediktor';
    document.querySelectorAll('#tpm sup').forEach(s => { s.title = 'Model Baseline: prediktor tidak mengalahkan klimatologi (LOYO) atau data < ' + MIN_N + ' tahun'; });
    const psm = $('psm'); if (psm) psm.textContent += ' Prediktor tambahan: ' + statusIdx() + '.';
    const cards = document.querySelectorAll('#cp .card');
    if (cards[1]) { const th = tahunan(kab), s = cards[1].querySelector('.s'); if (s && th.lo != null) s.textContent += ' · P10–P90 ' + fmt(th.lo, 0) + '–' + fmt(th.hi, 0) + ' mm'; }
    $('pnt').textContent = '† = Baseline (tanpa koreksi prediktor): tidak ada kombinasi ENSO/IOD/MJO/Kelvin/Rossby yang menurunkan galat leave-one-year-out ≥ ' +
      Math.round(C.MARGIN * 100) + '%, atau data < ' + MIN_N + ' tahun. Kategori memakai tercile klimatologi: Bawah/Atas Normal hanya bila peluangnya ≥ 40%.';
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
  window.PREDIKTOR = { C: C, nilai1: nilai1, serie: serie, fit2: fit2, annual: annual, kontribusi: kontribusi, setIdx: d => { IDX = d; iodC = null; resetFit(); }, htmlKotak: htmlKotak };
  muatIdx();
  if (typeof RAW !== 'undefined' && RAW && Object.keys(RAW).length && tahun != null) { resetFit(); render(); }
})();
