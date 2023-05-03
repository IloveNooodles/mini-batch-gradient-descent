# Forward Spec
Ada dua bagian: `case` dan `expect`.
Isi dari `case`:
* `model.input_size` (sudah jelas)
* `model.layers`, terdiri dari daftar objek beratribut berikut:
    * `number_of_neurons` (sudah jelas)
    * `activation_function` (nilai valid: `linear`, `relu`, `sigmoid`, `softmax`)
* `input`:
    * Dimensi 1 menandakan vektor ke-i. Ukurannya _arbitrary_.
    * Dimensi 2 menandakan isi suatu vektor. Ukurannya sesuai dengan `model.input_size`
* `weights`:
    * Dimensi 1 bersesuaian dengan _layer_ di `layers`.
    * Dimensi 2 berukuran banyak neuron pada lapisan sebelumnya + 1.
        * Khusus untuk _layer_ pertama: `model.input_size` + 1
        * Baris pertama adalah bias.
    * Dimensi 3 berukuran banyak _neuron_ pada lapisan yang bersesuaian.

Isi dari `expect`:
* `output`:
    * Ukuran dimensi 1 sesuai dengan ukuran dimensi 1 dari `input`.
    * Ukuran dimensi 2 sesuai dengan banyak _neuron_ pada lapisan terakhir.
* `max_sse`, nilai _sum of squares error_ yang dapat ditoleransi.
