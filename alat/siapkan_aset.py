"""Siapkan aset dashboard dari hasil analisis: model, ringkasan hasil (JSON), klip contoh, metadata ringkas.

Pakai: python alat/siapkan_aset.py <folder_dataset> <folder_cache_analisis>
Folder cache berisi emb_birdnet24.npy dan pred_*.npz dari notebook/skrip analisis.
"""
import sys, os, json, glob, shutil, numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

DATA, CACHE = sys.argv[1], sys.argv[2]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aset')
meta = pd.read_csv(os.path.join(DATA, 'metadata.csv'))
labels = sorted(meta['label'].unique()); y = meta['label'].map({l: i for i, l in enumerate(labels)}).to_numpy()
part = meta['bagian'].to_numpy(); tr, va, te = (np.where(part == b)[0] for b in ('latih', 'validasi', 'uji'))
rec = meta['id_rekaman'].to_numpy(); yt = y[te]

def macro_f1(a, b):
    f = []
    for c in range(10):
        tp = np.sum((a == c) & (b == c)); fp = np.sum((a != c) & (b == c)); fn = np.sum((a == c) & (b != c)); f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))
def f1_kelas(a, b):
    out = []
    for c in range(10):
        tp = np.sum((a == c) & (b == c)); fp = np.sum((a != c) & (b == c)); fn = np.sum((a == c) & (b != c)); out.append(float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0)
    return out

# 1) Model dashboard: BirdNET 2.4 + SVM, dilatih HANYA pada data latih (model yang sama dengan yang dievaluasi)
E = np.load(os.path.join(CACHE, 'emb_birdnet24.npy'))
model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', probability=True, random_state=0)).fit(E[tr], y[tr])
P = model.predict_proba(E[te]); pu = P.argmax(1)
print('model dashboard | uji macro-F1 (argmax peluang) %.4f | akurasi %.4f | top-3 %.4f' % (macro_f1(yt, pu), np.mean(pu == yt), np.mean([yt[i] in np.argsort(-P[i])[:3] for i in range(len(yt))])))
joblib.dump(dict(model=model, labels=labels), os.path.join(OUT, 'model_birdnet_svm.joblib'), compress=3)

# 2) Ringkasan hasil analisis
KEL = {'euclid': ('Jarak Euclid', 'Pencocokan pola'), 'dtw': ('MFCC + DTW', 'Pencocokan pola'), 'klasik': ('Fitur ringkasan + pengklasifikasi klasik', 'Fitur buatan tangan'),
       'cnn': ('CNN pada spektrogram', 'Jaringan saraf dari nol'), 'birdnet24': ('Embedding BirdNET 2.4 + SVM', 'Model suara burung terlatih'), 'perch2': ('Embedding Perch v2 + SVM', 'Model suara burung terlatih')}
urek = np.unique(rec[te]); idx_rek = [np.where(rec[te] == r)[0] for r in urek]
rng = np.random.default_rng(0); BOOT = [np.concatenate([idx_rek[i] for i in rng.integers(0, len(urek), len(urek))]) for _ in range(2000)]
rows = {}
for f in sorted(glob.glob(os.path.join(CACHE, 'pred_*.npz'))):
    n = os.path.basename(f)[5:-4]; d = np.load(f); pv, pt = d['val_pred'], d['test_pred']; b = np.array([macro_f1(yt[i], pt[i]) for i in BOOT])
    r = dict(model=n, kelompok=n.split('_')[0], val_f1=macro_f1(y[va], pv), uji_f1=macro_f1(yt, pt), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), uji_akurasi=float(np.mean(pt == yt)), pt=pt, boot=b)
    if 'test_proba' in d.files:
        Pm = d['test_proba']; pr = np.array([Pm[i].mean(0).argmax() for i in idx_rek]); yr = np.array([yt[i[0]] for i in idx_rek])
        r['rekaman_akurasi'] = float(np.mean(pr == yr)); r['rekaman_f1'] = macro_f1(yr, pr)
        r['top3'] = float(np.mean([yt[i] in np.argsort(-Pm[i])[:3] for i in range(len(yt))]))
    rows[n] = r
wakil = {}
for r in rows.values():
    if r['kelompok'] not in wakil or r['val_f1'] > wakil[r['kelompok']]['val_f1']: wakil[r['kelompok']] = r
urut = sorted(wakil.values(), key=lambda r: -r['val_f1'])
sel = meta['seleksi'].to_numpy()[te]
hasil = dict(
    labels=labels, n_klip=int(len(meta)), n_rekaman=int(meta['id_rekaman'].nunique()), n_perekam=int(meta['perekam'].nunique()),
    n_latih=int(len(tr)), n_validasi=int(len(va)), n_uji=int(len(te)), n_rekaman_uji=int(len(urek)),
    metode=[dict(model=r['model'], nama=KEL[r['kelompok']][0], keluarga=KEL[r['kelompok']][1], val_f1=r['val_f1'], uji_f1=r['uji_f1'], lo=r['lo'], hi=r['hi'], uji_akurasi=r['uji_akurasi'],
                 rekaman_akurasi=r.get('rekaman_akurasi'), top3=r.get('top3'), f1_spesies=f1_kelas(yt, r['pt']),
                 confusion=[[int(np.sum((yt == a) & (r['pt'] == b))) for b in range(10)] for a in range(10)],
                 kualitas={s: dict(n=int((sel == s).sum()), akurasi=float(np.mean(r['pt'][sel == s] == yt[sel == s]))) for s in ('aktivitas', 'cadangan')}) for r in urut],
    selisih=[dict(a=a['model'], b=b['model'], selisih=a['uji_f1'] - b['uji_f1'], lo=float(np.percentile(a['boot'] - b['boot'], 2.5)), hi=float(np.percentile(a['boot'] - b['boot'], 97.5))) for a, b in zip(urut, urut[1:])],
    semua_model=[dict(model=r['model'], kelompok=KEL[r['kelompok']][0], val_f1=r['val_f1'], uji_f1=r['uji_f1'], lo=r['lo'], hi=r['hi'], uji_akurasi=r['uji_akurasi']) for r in sorted(rows.values(), key=lambda r: -r['val_f1'])],
    klip_uji_per_spesies=[int((yt == c).sum()) for c in range(10)],
)
# Kalibrasi sederhana model dashboard: seberapa sering tebakan benar pada tiap rentang keyakinan (data uji)
yakin = P.max(1)
hasil['kalibrasi'] = [dict(lo=lo, hi=hi, n=int(((yakin >= lo) & (yakin < hi)).sum()), akurasi=float(np.mean(pu[(yakin >= lo) & (yakin < hi)] == yt[(yakin >= lo) & (yakin < hi)])))
                      for lo, hi in ((0.0, 0.45), (0.45, 0.70), (0.70, 1.01))]
hasil['model_dashboard'] = dict(akurasi=float(np.mean(pu == yt)), top3=float(np.mean([yt[i] in np.argsort(-P[i])[:3] for i in range(len(yt))])))
print('kalibrasi', hasil['kalibrasi'])
json.dump(hasil, open(os.path.join(OUT, 'hasil.json'), 'w'), indent=1)

# 3) Metadata ringkas (untuk halaman data dan peta)
kol = ['spesies', 'nama_ilmiah', 'label', 'bagian', 'id_rekaman', 'seleksi', 'snr_db', 'jenis_suara', 'kualitas_xc', 'perekam', 'lokasi', 'lat', 'lon', 'tanggal', 'lisensi', 'halaman_sumber']
meta[kol].to_csv(os.path.join(OUT, 'metadata.csv'), index=False)

# 4) Klip contoh: dari data UJI, lolos penyaring suara, lisensi tanpa "ND", tiga rekaman berbeda per spesies dengan SNR tertinggi.
#    Dipilih TANPA melihat benar/salahnya tebakan model.
shutil.rmtree(os.path.join(OUT, 'contoh'), ignore_errors=True); os.makedirs(os.path.join(OUT, 'contoh'))
uji = meta.iloc[te].copy(); uji['idx'] = te; uji['pos'] = np.arange(len(te))
calon = uji[(uji['seleksi'] == 'aktivitas') & (~uji['lisensi'].str.contains('-nd', case=False))]
contoh, emb = [], []
for lab in labels:
    g = calon[calon['label'] == lab].sort_values('snr_db', ascending=False).drop_duplicates('id_rekaman').head(3)
    for k, (_, r) in enumerate(g.iterrows()):
        nama = f'{lab}_{k+1}.wav'; shutil.copy(os.path.join(DATA, r['berkas']), os.path.join(OUT, 'contoh', nama))
        contoh.append(dict(berkas=nama, label=lab, spesies=r['spesies'], id_rekaman=r['id_rekaman'], perekam=r['perekam'], lokasi=r['lokasi'], lisensi=r['lisensi'], halaman_sumber=r['halaman_sumber'], mulai_ms=int(r['mulai_ms'])))
        emb.append(E[r['idx']])
json.dump(contoh, open(os.path.join(OUT, 'contoh', 'contoh.json'), 'w'), indent=1); np.save(os.path.join(OUT, 'contoh', 'embedding.npy'), np.stack(emb).astype(np.float32))
pc = model.predict(np.stack(emb)); print('klip contoh:', len(contoh), '| ditebak benar oleh model:', int(sum(labels[p] == c['label'] for p, c in zip(pc, contoh))))
print('ukuran aset (MB):', {f: round(os.path.getsize(os.path.join(OUT, f)) / 1e6, 2) for f in os.listdir(OUT) if os.path.isfile(os.path.join(OUT, f))})
