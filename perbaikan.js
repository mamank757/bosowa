/* perbaikan.js — muat SETELAH skrip utama index.html (satu baris sebelum </body>).
   Isi: BoM keluar dari proyeksi, tahunan() aktual+sisa hari, kategori tercile,
   uji ENSO lebih ketat (|t| >= 2), label indeks/sumber, penanda bulan tanpa koreksi ENSO,
   peringatan kabupaten berdata pendek. */
(function () {
  'use strict';
  const T_BARU = 2.0;   // ambang |t| (sebelumnya 1,65); validasi: Mei (t=1,9) tidak stabil, Feb/Apr/Agu-Okt tetap lolos

  /* 1. cv klimatologi + ambang uji lebih ketat */
  const _fitPairs = fitPairs;
  fitPairs = function (P, m0) {
    const R = _fitPairs(P, m0);
    let cv0 = .5;
    if (P.length >= 3) {
      const mu = P.reduce((s, p) => s + p.y, 0) / P.length;
      const sd = Math.sqrt(P.reduce((s, p) => s + (p.y - mu) ** 2, 0) / (P.length - 1));
      if (mu > 0) cv0 = Math.min(1.5, Math.max(.2, sd / mu));
    }
    R.cv0 = cv0;
    if (R.ok && Math.abs(R.t) < T_BARU) { R.ok = false; R.a = Math.log(Math.max(R.mean, 1e-6)); R.b = 0; R.cv = cv0; }
    return R;
  };

  /* 2. BoM TIDAK dipakai dalam proyeksi (indeks berbeda dari OISST yang dipakai untuk kalibrasi) */
  ensoAutoFor = function (list) {
    const kn = list.map(m => { const k = key(m);
      return ENS.obs[k] !== undefined ? { v: ENS.obs[k], s: 'obs' } : ENS.mtd[k] !== undefined ? { v: ENS.mtd[k], s: 'mtd' } :
        ENS.fcst[k] !== undefined ? { v: ENS.fcst[k], s: 'fc' } : ENS.fauto[k] !== undefined ? { v: ENS.fauto[k], s: 'stat' } : null; });
    let L = -1; kn.forEach((k, i) => { if (k) L = i; });
    return list.map((m, i) => kn[i] || { v: Math.round((L >= 0 ? kn[L].v * Math.exp(-Math.max(0, i - L) / TAU) : 0) * 100) / 100, s: 'pro' });
  };

  /* 3. Peluang berbasis tercile klimatologi: tanpa sinyal ENSO => 33/33/33 (bukan 51/16/33).
        Kategori BN/AN hanya bila peluangnya >= 40%. Ketidakpastian tidak pernah melebihi klimatologi. */
  compute = function (id, list, ev) {
    const map = lagMap();
    return list.map((m, i) => {
      const f = fit(id, m.mo), e = ev[i];
      if (!f) return { m: m, e: e, none: true };
      const lead = Math.max(0, m.y * 12 + m.mo - 1 - NOWZ), x = lagE(m.y, m.mo, map), xe = x === null ? e : x;
      const att = 1 - ATT * Math.min(1, Math.max(0, lead - 1) / 11), p = predict(f, xe, att), Nm = f.mean;
      const cv0 = f.cv0 || f.cv, cvF = f.ok ? Math.min(cv0, Math.sqrt(f.cv ** 2 + (.04 * lead) ** 2)) : cv0;
      const ds = dist(p.mu, cvF);
      let pb = 0, pa = 0;
      if (ds.a !== null) {
        const a0 = 1 / (cv0 * cv0), th0 = Nm / a0, lo = gq(1 / 3, a0, th0), hi = gq(2 / 3, a0, th0);
        pb = Math.max(0, Math.min(1, gp(ds.a, lo / ds.th))); pa = Math.max(0, Math.min(1, 1 - gp(ds.a, hi / ds.th)));
      }
      const pn = Math.max(0, 1 - pb - pa), cat = pb >= .4 && pb >= pa ? 'bn' : pa >= .4 ? 'an' : 'n';
      return { m: m, e: e, F: p.F, n: f.n, t: f.t, fitOK: f.ok, Nm: Nm, mean: p.mu, p10: ds.p10, p90: ds.p90, pb: pb, pn: pn, pa: pa, cat: cat };
    });
  };

  /* 4. Estimasi tahunan = aktual + proyeksi untuk hari yang BELUM tercatat (bukan max) */
  tahunan = function (id) {
    const ms = compute(id, CAL, ensoAutoFor(CAL).map(r => r.v)), rows = rowsOf(id, REPORT_YEAR);
    let tb = 0, tp = 0, nO = 0, n = 0;
    ms.forEach((c, i) => {
      if (c.none) return; n++; tb += c.Nm;
      const r = rows[i];
      if (r && r.lengkap && valid(r)) { tp += r.mm; nO++; }
      else if (r && valid(r) && r.hari > 0) tp += r.mm + c.mean * Math.max(0, r.hari_bulan - r.hari) / r.hari_bulan;
      else tp += c.mean;
    });
    return { tb: tb, tp: tp, nO: nO, n: n };
  };

  /* 5. Label & peringatan (dijalankan setelah setiap render) */
  const ganti = (el, a, b) => { if (el && el.innerHTML.indexOf(a) > -1) el.innerHTML = el.innerHTML.split(a).join(b); };
  function pasca() {
    ganti($('enso'), 'Prakiraan BoM', 'BoM (Niño-3.4 relatif)');
    ganti($('enso'), ' · input manual', ' · otomatis dari data mingguan BoM');
    document.querySelectorAll('.leg span').forEach(s => ganti(s, 'Prakiraan BoM (input manual)', 'BoM Niño-3.4 relatif (otomatis, bulan belum dirilis NOAA)'));
    document.querySelectorAll('#cards .card').forEach(c => {
      const k = c.querySelector('.k'), s = c.querySelector('.s');
      if (k && s && /Dampak ENSO/.test(k.textContent) && /Korelasi/.test(s.textContent)) s.textContent = 'Korelasi ONI–hujan Jul–Nov, tanpa jeda';
    });
    if (mode !== 'proy') return;

    const awas = [];
    if (ENS.kind === 'RONI') awas.push('Sumber cadangan RONI (indeks relatif) sedang dipakai, bukan OISST; hasil dapat berbeda dari biasanya.');
    if (ENS.mode === 'cache') awas.push('Data ENSO berasal dari cache lokal (sumber langsung gagal dimuat).');
    $('psm').textContent = 'Indeks: anomali Niño-3.4 bulanan' + (ENS.kind ? ' (' + ENS.kind + ')' : '') +
      '. Data BoM tidak dipakai dalam proyeksi, hanya melengkapi grafik ENSO.' + (awas.length ? ' ⚠️ ' + awas.join(' ') : '');

    const cards = document.querySelectorAll('#cp .card');
    if (cards[0]) cards[0].querySelector('.s').textContent = 'Jumlah rata-rata ' + tahunan(kab).n + ' bulan kalender ' + REPORT_YEAR;
    if (cards[2]) { const e = ensoAutoFor(CAL)[now.getMonth()].v; cards[2].querySelector('.s').textContent = 'Niño-3.4 ' + sg(e) + ' (' + (ENS.kind || 'bulanan') + ', bukan ONI)'; }

    const rows = document.querySelectorAll('#tpm tr');
    PM.forEach((x, i) => {
      if (x.none || x.fitOK || !rows[i] || !rows[i].children[6]) return;
      rows[i].children[6].insertAdjacentHTML('beforeend', '<sup title="Tanpa koreksi ENSO: proyeksi = rata-rata historis">†</sup>');
    });

    const pendek = KAB.filter(k => { let ok = 0; for (let mo = 1; mo <= 12; mo++) { const f = fit(k.id, mo); if (f && f.n >= MIN_N) ok++; } return ok === 0 && Object.keys(A[k.id].years).length; });
    const tr = $('tsb').children;
    KAB.forEach((k, j) => { if (pendek.indexOf(k) > -1 && tr[j] && tr[j].children.length >= 7) tr[j].lastElementChild.textContent += ' · data < 8 th'; });
    $('pnt').textContent = '† = tanpa koreksi ENSO (hubungan tidak signifikan, |t| < ' + T_BARU.toString().replace('.', ',') + ', atau data < ' + MIN_N + ' tahun): proyeksi = rata-rata historis. ' +
      'Kategori memakai tercile klimatologi: Bawah/Atas Normal hanya bila peluangnya ≥ 40%.' +
      (pendek.length ? ' Belum ada koreksi ENSO sama sekali (data terlalu pendek): ' + pendek.map(k => k.nm).join(', ') + '; hasilnya hanya rata-rata historis.' : '');
  }
  const _render = render;
  render = function () { _render(); try { pasca(); } catch (e) { console.warn('perbaikan.js:', e); } };

  const _bp = blokPDF;
  blokPDF = function () { return _bp().split('prakiraan BoM').join('data BoM (otomatis)'); };

  /* 6. Bila halaman sudah sempat tampil dari cache sebelum file ini dimuat, hitung ulang sekali */
  if (typeof RAW !== 'undefined' && RAW && Object.keys(RAW).length && tahun != null) { proses(RAW); render(); }
})();
