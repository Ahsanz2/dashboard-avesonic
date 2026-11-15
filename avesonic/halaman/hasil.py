import numpy as np
import plotly.graph_objects as go
import streamlit as st
from avesonic import tema, mesin
from avesonic.tema import angka, persen

GAGASAN = {
    "perch2": "Model Perch v2 (Google), yang sudah belajar dari suara ribuan spesies, membuat \"sidik suara\" tiap klip. Model sederhana lalu diajari membedakan 10 spesies ini.",
    "birdnet24": "Sama seperti di atas, memakai model BirdNET 2.4 (Cornell dan TU Chemnitz). Ini model yang dipakai di halaman Coba Tebak.",
    "cnn": "Jaringan saraf melihat potret suara (spektrogram) seperti melihat gambar, dan belajar dari nol hanya dari data ini.",
    "klasik": "Tiap klip diringkas menjadi 476 angka statistik (tinggi nada, kekasaran, dan sebagainya), lalu model statistik sederhana belajar membedakannya.",
    "dtw": "Klip baru dibandingkan langsung dengan ribuan klip contoh. Sumbu waktu boleh \"ditarik-ulur\" agar pola yang sama tetapi beda tempo tetap cocok.",
    "euclid": "Seperti DTW, tetapi kaku: dibandingkan titik demi titik pada waktu yang sama, tanpa tarik-ulur.",
}


def _kunci(m):
    return m["model"].split("_")[0]


def tampil():
    h = mesin.hasil(); met = h["metode"]; labels = h["labels"]
    terbaik, kedua = met[0], met[1]
    st.markdown('<div class="eyebrow">Hasil analisis</div>', unsafe_allow_html=True)
    st.title("Enam cara menebak, satu ujian yang sama")
    st.markdown(f"Keenam metode diuji pada **{angka(h['n_uji'])} klip dari {h['n_rekaman_uji']} rekaman** yang tidak pernah dipakai saat melatih. "
                f"Pemenangnya jelas: metode yang memanfaatkan model suara burung terlatih menebak benar sekitar **{persen(terbaik['uji_akurasi'])}** klip.")

    # ------------------------------------------------------------ perbandingan
    tema.judul_bagian("Perbandingan", "Metode mana yang paling tepat?")
    ukuran = st.segmented_control("Ukuran", ["Macro-F1", "Akurasi per klip", "Akurasi per rekaman"], default="Macro-F1", label_visibility="collapsed") or "Macro-F1"
    if ukuran == "Macro-F1":
        d = met; nilai = [m["uji_f1"] for m in d]
        fig = tema.batang_h([m["nama"] for m in d], nilai, [angka(v, 2) for v in nilai],
                            [f"<b>{m['nama']}</b><br>{m['keluarga']}<br>macro-F1 {angka(m['uji_f1'], 3)} (rentang 95%: {angka(m['lo'], 2)}–{angka(m['hi'], 2)})<br>akurasi per klip {persen(m['uji_akurasi'], 1)}" for m in d],
                            maks=1.0, bawah=[m["lo"] for m in d], atas=[m["hi"] for m in d], judul_x="macro-F1 di data uji (0 sampai 1)", sorot=[0, 1])
        ket = "**Macro-F1** adalah rata-rata nilai ketepatan tiap spesies (0 sampai 1), sehingga spesies yang datanya sedikit sama pentingnya dengan yang banyak. Garis hitam menunjukkan rentang ketidakpastian 95%."
    elif ukuran == "Akurasi per klip":
        d = met; nilai = [m["uji_akurasi"] for m in d]
        fig = tema.batang_h([m["nama"] for m in d], nilai, [persen(v) for v in nilai], [f"<b>{m['nama']}</b><br>{persen(m['uji_akurasi'], 1)} dari {angka(h['n_uji'])} klip uji ditebak benar" for m in d],
                            maks=1.0, judul_x="bagian klip uji yang ditebak benar", sorot=[0, 1])
        fig.update_xaxes(tickformat=".0%")
        fig.add_vline(x=0.1, line=dict(color=tema.REDUP, width=1, dash="dot"), annotation_text="tebak acak 10%", annotation_position="top", annotation_font=dict(color=tema.REDUP, size=11))
        ket = "**Akurasi per klip**: dari semua klip 3 detik di data uji, berapa bagian yang spesiesnya ditebak benar."
    else:
        d = [m for m in met if m["rekaman_akurasi"] is not None]; nilai = [m["rekaman_akurasi"] for m in d]
        fig = tema.batang_h([m["nama"] for m in d], nilai, [persen(v) for v in nilai], [f"<b>{m['nama']}</b><br>{persen(m['rekaman_akurasi'], 1)} dari {h['n_rekaman_uji']} rekaman uji ditebak benar<br>(per klip: {persen(m['uji_akurasi'], 1)})" for m in d],
                            maks=1.0, judul_x="bagian rekaman uji yang ditebak benar", sorot=[0, 1])
        fig.update_xaxes(tickformat=".0%")
        ket = "**Akurasi per rekaman**: tebakan semua klip dari satu rekaman digabung menjadi satu jawaban. Tidak tersedia untuk DTW dan Euclid karena keduanya tidak menghasilkan nilai peluang."
    st.plotly_chart(fig, width="stretch", config=tema.KONFIG)
    st.caption(ket)

    with st.expander("Apa bedanya keenam metode itu? (penjelasan sederhana)"):
        for m in met:
            st.markdown(f"**{m['nama']}** · _{m['keluarga']}_  \n{GAGASAN[_kunci(m)]}")

    # ------------------------------------------------------------ selisih
    tema.judul_bagian("Kepastian", "Apakah selisihnya sungguhan, atau kebetulan?",
                      "Selisih skor antara dua metode yang peringkatnya berurutan. Jika garisnya tidak menyentuh angka nol, selisih itu dapat dipercaya.")
    nama = {m["model"]: m["nama"] for m in met}
    S = h["selisih"]; yl = [f"{nama[s['a']].split(' + ')[0].replace('Embedding ', '')}  vs  {nama[s['b']].split(' + ')[0].replace('Embedding ', '')}" for s in S]
    fig = go.Figure(go.Scatter(
        x=[s["selisih"] for s in S], y=yl, mode="markers", marker=dict(size=11, color=tema.DATA, line=dict(color="#ffffff", width=2)),
        error_x=dict(type="data", symmetric=False, array=[s["hi"] - s["selisih"] for s in S], arrayminus=[s["selisih"] - s["lo"] for s in S], color=tema.DATA, thickness=2, width=0),
        customdata=[[angka(s["selisih"], 3), angka(s["lo"], 3), angka(s["hi"], 3), "selisih nyata" if s["lo"] > 0 else "belum bisa dibedakan"] for s in S],
        hovertemplate="<b>%{y}</b><br>selisih macro-F1 %{customdata[0]}<br>rentang 95%: %{customdata[1]} sampai %{customdata[2]}<br>%{customdata[3]}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=tema.TINTA, width=1))
    for s, y in zip(S, yl):
        fig.add_annotation(x=s["hi"], y=y, text="nyata" if s["lo"] > 0 else "<b>belum bisa dibedakan</b>", showarrow=False, xanchor="left", xshift=10, font=dict(size=12, color=tema.TINTA2 if s["lo"] > 0 else tema.TINTA))
    tema.gaya(fig, tinggi=300); fig.update_yaxes(autorange="reversed", showgrid=False, ticksuffix="  "); fig.update_xaxes(title_text="selisih macro-F1 (metode pertama dikurangi metode kedua)", range=[-0.06, 0.36])
    st.plotly_chart(fig, width="stretch", config=tema.KONFIG)
    s0 = S[0]
    tema.catatan(f"<b>{terbaik['nama'].split(' + ')[0].replace('Embedding ', '')} dan {kedua['nama'].split(' + ')[0].replace('Embedding ', '')} praktis seri.</b> Selisihnya hanya {angka(s0['selisih'], 3)} dan rentangnya melewati nol "
                 f"({angka(s0['lo'], 3)} sampai {angka(s0['hi'], 3)}). Selisih antarperingkat lainnya nyata.")

    # ------------------------------------------------------------ per spesies
    tema.judul_bagian("Per spesies", "Spesies mana yang mudah, mana yang sulit?")
    pilih = st.selectbox("Metode", [m["nama"] for m in met], index=0)
    m = next(x for x in met if x["nama"] == pilih)
    cm = np.array(m["confusion"]); f1 = np.array(m["f1_spesies"]); idn = [mesin.nama_id(l) for l in labels]
    a, b = st.columns([5, 6], gap="medium")
    with a:
        o = np.argsort(-f1)
        fig = tema.batang_h([idn[i] for i in o], [float(f1[i]) for i in o], [angka(f1[i], 2) for i in o],
                            [f"<b>{idn[i]}</b><br><i>{mesin.SPESIES[labels[i]][2]}</i><br>F1 {angka(f1[i], 3)} · {h['klip_uji_per_spesies'][i]} klip uji · {cm[i, i]} ditebak benar" for i in o],
                            tinggi=470, maks=1.12, judul_x="F1 per spesies (0 sampai 1)")
        st.plotly_chart(fig, width="stretch", config=tema.KONFIG)
    with b:
        norm = cm / cm.sum(1, keepdims=True)
        teks = [[str(v) if v else "" for v in r] for r in cm]
        fig = go.Figure(go.Heatmap(z=norm, x=idn, y=idn, zmin=0, zmax=1, colorscale=[[i / 5, c] for i, c in enumerate(tema.RAMP)], showscale=False, xgap=2, ygap=2,
                                   text=teks, texttemplate="%{text}", textfont=dict(size=11),
                                   customdata=[[[idn[i], idn[j], int(cm[i, j]), persen(norm[i, j])] for j in range(10)] for i in range(10)],
                                   hovertemplate="sebenarnya <b>%{customdata[0]}</b><br>ditebak <b>%{customdata[1]}</b><br>%{customdata[2]} klip (%{customdata[3]} dari spesies ini)<extra></extra>"))
        tema.gaya(fig, tinggi=470); fig.update_yaxes(autorange="reversed", showgrid=False, title_text="spesies sebenarnya", tickfont=dict(size=11)); fig.update_xaxes(showgrid=False, tickangle=-40, title_text="tebakan model", tickfont=dict(size=11, color=tema.TINTA2))
        st.plotly_chart(fig, width="stretch", config=tema.KONFIG)
    st.caption("Kiri: nilai F1 tiap spesies. Kanan: tabel silang; tiap baris adalah spesies sebenarnya, tiap kolom tebakan model. Kotak gelap di diagonal berarti tebakan benar; kotak di luar diagonal adalah salah tebak.")
    luar = cm.copy(); np.fill_diagonal(luar, 0); tiga = np.dstack(np.unravel_index(np.argsort(-luar, axis=None)[:3], luar.shape))[0]
    tema.catatan("<b>Paling sering tertukar:</b> " + " · ".join(f"{idn[i]} ditebak sebagai {idn[j]} ({luar[i, j]} klip)" for i, j in tiga) + ".")

    # ------------------------------------------------------------ kualitas
    tema.judul_bagian("Kualitas klip", "Klip yang jernih jauh lebih mudah ditebak")
    q = m["kualitas"]
    tema.ubin([
        (persen(q["aktivitas"]["akurasi"]), "benar pada klip \"jelas\"", f"{angka(q['aktivitas']['n'])} klip uji"),
        (persen(q["cadangan"]["akurasi"]), "benar pada klip \"cadangan\"", f"{angka(q['cadangan']['n'])} klip uji"),
        (persen(q["aktivitas"]["akurasi"] - q["cadangan"]["akurasi"]).replace("%", " poin"), "selisih", "untuk metode yang dipilih di atas"),
    ])
    st.caption("Klip \"cadangan\" adalah klip yang suaranya terus-menerus atau sangat sunyi sehingga tidak lolos penyaring otomatis. Membuang klip semacam ini kemungkinan menaikkan hasil; hal itu belum diuji.")

    # ------------------------------------------------------------ DTW
    tema.judul_bagian("Peragaan", "Bagaimana DTW \"menarik-ulur\" waktu?",
                      "Dua burung sejenis jarang berkicau dengan tempo yang persis sama. DTW (Dynamic Time Warping) mencari cara terbaik menyejajarkan dua suara dengan memelarkan atau memampatkan waktunya. Pilih dua klip untuk melihatnya.")
    daftar, _ = mesin.contoh()
    opsi = list(range(len(daftar))); fmt = lambda i: f"{mesin.nama_id(daftar[i]['label'])} · klip {daftar[i]['berkas'].rsplit('_', 1)[1][0]}"
    ia_awal = next(i for i, c in enumerate(daftar) if c["label"] == "pygmy_cupwing")
    c1, c2 = st.columns(2)
    with c1:
        ia = st.selectbox("Klip A", opsi, index=ia_awal, format_func=fmt); st.audio(mesin.jalur_contoh(daftar[ia]["berkas"]))
    with c2:
        ib = st.selectbox("Klip B", opsi, index=ia_awal + 1, format_func=fmt); st.audio(mesin.jalur_contoh(daftar[ib]["berkas"]))
    D, jalur, d_dtw, d_euc = mesin.dtw_contoh(daftar[ia]["berkas"], daftar[ib]["berkas"])
    t = np.arange(D.shape[0]) * 0.02
    fig = go.Figure(go.Heatmap(z=D.T, x=t, y=t, colorscale=[[i / 5, c] for i, c in enumerate(tema.RAMP[::-1])], showscale=False, hovertemplate="A detik %{x:.2f} · B detik %{y:.2f}<br>jarak %{z:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, t[-1]], y=[0, t[-1]], mode="lines", line=dict(color="#ffffff", width=1.5, dash="dot"), name="tanpa tarik-ulur (Euclid)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[i * 0.02 for i, _ in jalur], y=[j * 0.02 for _, j in jalur], mode="lines", line=dict(color=tema.EMAS, width=3), name="jalur DTW", hoverinfo="skip"))
    tema.gaya(fig, tinggi=440, legenda=True); fig.update_xaxes(title_text="waktu klip A (detik)", showgrid=False, constrain="domain"); fig.update_yaxes(title_text="waktu klip B (detik)", showgrid=False, scaleanchor="x")
    g1, g2 = st.columns([6, 5], gap="medium")
    with g1:
        st.plotly_chart(fig, width="stretch", config=tema.KONFIG)
    with g2:
        sama = daftar[ia]["label"] == daftar[ib]["label"]
        tema.ubin([(angka(d_euc, 1), "jarak tanpa tarik-ulur", "rata-rata per langkah (Euclid)"), (angka(d_dtw, 1), "jarak dengan DTW", "rata-rata per langkah di jalur terbaik")])
        st.markdown(f"Kedua klip ini berasal dari **spesies yang {'sama' if sama else 'berbeda'}**. Warna gelap berarti kedua suara mirip pada pasangan waktu itu. "
                    f"Garis emas adalah penyejajaran terbaik yang ditemukan DTW; makin jauh ia melengkung dari garis putus-putus, makin besar perbedaan temponya.")
        st.caption("DTW selalu menghasilkan jarak yang sama atau lebih kecil daripada Euclid. Yang berguna untuk menebak spesies adalah apakah jarak ke klip sejenis menjadi lebih kecil daripada jarak ke klip spesies lain. Coba bandingkan dengan memilih klip B dari spesies lain.")
