from flask import Flask, render_template, request
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load("model_tbc.pkl")

# Nama fitur
feature_names = [
    "Usia",
    "Jenis Kelamin",
    "Riwayat Kontak TB",
    "Status Merokok",
    "Kepadatan Hunian",
    "Ventilasi Rumah",
    "Konsumsi Alkohol",
    "Batuk > 2 Minggu",
    "Demam Berkepanjangan",
    "Keringat Malam",
    "Penurunan Berat Badan",
    "Nafsu Makan Menurun",
    "Sesak Napas",
    "Nyeri Dada",
    "Batuk Berdarah"
]


# ==========================
# HALAMAN UTAMA
# ==========================
@app.route("/")
def index():
    return render_template("index.html")


# ==========================
# PREDIKSI
# ==========================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = np.array([[
            int(request.form["usia"]),
            int(request.form["jenis_kelamin"]),
            int(request.form["riwayat_kontak"]),
            int(request.form["status_merokok"]),
            int(request.form["kepadatan_hunian"]),
            int(request.form["ventilasi_rumah"]),
            int(request.form["konsumsi_alkohol"]),
            int(request.form["batuk_2_minggu"]),
            int(request.form["demam"]),
            int(request.form["keringat_malam"]),
            int(request.form["penurunan_bb"]),
            int(request.form["nafsu_makan"]),
            int(request.form["sesak_napas"]),
            int(request.form["nyeri_dada"]),
            int(request.form["batuk_berdarah"])
        ]])

        # ==========================
        # PREDIKSI MODEL
        # ==========================
        prediction = model.predict(data)[0]

        probability = round(
            model.predict_proba(data)[0][1] * 100,
            2
        )

        # ==========================
        # HASIL KLASIFIKASI
        # ==========================
        if prediction == 1:
            hasil = "Terindikasi Tuberculosis (TBC)"
            badge = "danger"
        else:
            hasil = "Tidak Terindikasi Tuberculosis (TBC)"
            badge = "success"

        # ==========================
        # WARNA PROGRESS BAR
        # ==========================
        if probability >= 80:
            progress_color = "bg-danger"
        elif probability >= 50:
            progress_color = "bg-warning"
        else:
            progress_color = "bg-success"

        # ==========================
        # LEVEL RISIKO
        # ==========================
        if probability >= 80:
            risk_level = "Risiko Tinggi"
        elif probability >= 50:
            risk_level = "Risiko Sedang"
        else:
            risk_level = "Risiko Rendah"

        # ==========================
        # INTERPRETASI SISTEM
        # ==========================
        if probability >= 80:
            interpretasi = (
                "Pasien menunjukkan kombinasi faktor risiko "
                "dan gejala yang sangat kuat mengarah pada "
                "Tuberkulosis (TBC). Pemeriksaan lanjutan "
                "sangat direkomendasikan."
            )

        elif probability >= 50:
            interpretasi = (
                "Pasien memiliki beberapa faktor risiko "
                "yang berkaitan dengan Tuberkulosis (TBC). "
                "Disarankan melakukan pemeriksaan klinis "
                "lebih lanjut."
            )

        else:
            interpretasi = (
                "Probabilitas Tuberkulosis (TBC) relatif "
                "rendah berdasarkan data yang dimasukkan. "
                "Namun pemantauan kesehatan tetap dianjurkan."
            )

        # ==========================
        # REPORT ID
        # ==========================
        report_id = datetime.now().strftime(
            "TBC-%Y%m%d-%H%M%S"
        )

        # ==========================
        # RINGKASAN PASIEN
        # ==========================
        data_pasien = {
            "Usia": f"{request.form['usia']} Tahun",
            "Jenis Kelamin": (
                "Laki-laki"
                if request.form["jenis_kelamin"] == "0"
                else "Perempuan"
            ),
            "Riwayat Kontak TB": (
                "Ya"
                if request.form["riwayat_kontak"] == "1"
                else "Tidak"
            ),
            "Status Merokok": (
                "Ya"
                if request.form["status_merokok"] == "1"
                else "Tidak"
            ),
            "Kepadatan Hunian": (
                "Tinggi"
                if request.form["kepadatan_hunian"] == "1"
                else "Rendah"
            ),
            "Ventilasi Rumah": (
                "Buruk"
                if request.form["ventilasi_rumah"] == "1"
                else "Baik"
            ),
            "Konsumsi Alkohol": (
                "Ya"
                if request.form["konsumsi_alkohol"] == "1"
                else "Tidak"
            )
        }

        # ==========================
        # FEATURE IMPORTANCE
        # ==========================
        feature_importance = []
        top_factor = "-"

        if hasattr(model, "feature_importances_"):

            importance = model.feature_importances_

            feature_importance = sorted(
                zip(feature_names, importance),
                key=lambda x: x[1],
                reverse=True
            )

            # Ambil 5 faktor terbesar
            feature_importance = feature_importance[:5]

            if len(feature_importance) > 0:
                top_factor = feature_importance[0][0]

        # ==========================
        # WAKTU PREDIKSI
        # ==========================
        waktu_prediksi = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        return render_template(
            "result.html",
            hasil=hasil,
            probability=probability,
            badge=badge,
            progress_color=progress_color,
            data_pasien=data_pasien,
            feature_importance=feature_importance,
            waktu_prediksi=waktu_prediksi,
            risk_level=risk_level,
            interpretasi=interpretasi,
            report_id=report_id,
            top_factor=top_factor,
            error=None
        )

    except Exception as e:

        return render_template(
            "result.html",
            hasil="Terjadi Kesalahan",
            probability=0,
            badge="secondary",
            progress_color="bg-secondary",
            data_pasien={},
            feature_importance=[],
            waktu_prediksi="-",
            risk_level="-",
            interpretasi="-",
            report_id="-",
            top_factor="-",
            error=str(e)
        )


# ==========================
# RUN FLASK
# ==========================
if __name__ == "__main__":
    app.run(debug=True)