# Backward Spec
Ada dua bagian: `case` dan `expect`.
Isi dari `case`:
* `model.input_size` (sudah jelas)
* `model.layers`:
    * `number_of_neurons` (sudah jelas)
    * `activation_function` (nilai valid: `linear`, `relu`, `sigmoid`, `softmax`)
* `input`:
    * Dimensi 1 menandakan vektor ke-i. Ukurannya _arbitrary_.
    * Dimensi 2 menandakan isi suatu vektor. Ukurannya sesuai dengan `model.input_size`
* `initial_weights`:
    * Dimensi 1 bersesuaian dengan _layer_ di `layers`.
    * Dimensi 2 berukuran banyak neuron pada lapisan sebelumnya + 1.
        * Khusus untuk _layer_ pertama: `model.input_size` + 1
        * Baris pertama adalah bias.
    * Dimensi 3 berukuran banyak _neuron_ pada lapisan yang bersesuaian.
* `target`:
    * Ukuran dimensi 1 sesuai dengan ukuran dimensi 1 dari `input`.
    * Ukuran dimensi 2 sesuai dengan banyak _neuron_ pada lapisan terakhir.
* `learning_parameters`:
    * `learning_rate` (sudah jelas)
    * `batch_size` (sudah jelas)
    * `max_iteration` (sudah jelas)
    * `error_threshold`:
        * Perhitungan eror sesuai dengan fungsi aktivasi pada lapisan terakhir.
        * Fungsi log jika mengacu pada spesifikasi berbasis e (ekuivalen dengan ln(x)).

Isi dari `expect`:
* `stopped_by`
    * Hanya ada dua nilai yang valid: `max_iteration` dan `error_threshold`.
    * Jika bernilai `max_iteration`, diharapkan pembelajaran berhenti karena banyak iterasinya mencapai maksimum.
    * Jika bernilai `error_threshold`, diharapkan pembelajaran berhenti karena rata-rata nilai eror yang diperoleh pada iterasi terakhir lebih kecil atau sama dengan error threshold.
* `final_weights`:
    * Jika atribut ini dinyatakan, kesamaan _weights_ dicek setelah semua iterasi pembelajaran dilakukan.
    * Spesifikasi dimensi sama dengan `case.initial_weights`.
    * Maksimum SSE (_sum of squares error_) yang dapat diterima adalah 10^-8.
