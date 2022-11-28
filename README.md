# Laporan Projek Machine Learning Terapan - Sistem Rekomendasi - Mohammad Rafdi
## Domain Proyek
Film merupakan sebuah gambar yang bergerak, terdiri dari beberapa gambar pada setiap detiknya. Film diartikan sebagai hasil budaya dan alat ekspresi kesenian.
Film memiliki banyak fungsi seperti memberikan edukasi, hiburan dan dokumentasi suatu kejadian. Perkembangan manfaat menonton film membuat film dikategerikan menjadi beberapa genre.
genre merupakan sebutan untuk membedakan jenis film.film bisa bersifat fiksi atau kisah nyata ataupun campuran keduanya. Walaupun ratusan film dibuat setiap tahunnya tetapi hanya sedikit film hanya menggunakan satu genre kebanyakan menggabungkan dua genre atau lebih[1].


Seiring pesatnya perkembangan digital di bidang multimedia aplikasi pemutaran film telah tersedia.
Sistem rekomendasi merupakan sebuah sistem pendukung keputusan yang membantu pengguna aplikasi musik dapat menerima rekomendasi musik musik yang sesuai dengan kriterianya. Sistem rekomendasi film ini mengambil informasi informasi seperti film yang pernah ditonton, genre kesukaan, film yang diputar berulang ulang, dan masih banyak yang lainnya. Maka dari itu penulis akan membuat sebuah sistem rekomendasi yang dapat membantu manusia mendapat rekomendasi film kesukaannya tanpa harus lelah mencari dan berpikir film yang harus diputar selanjutnya. 

## Business Understanding
Para penyedia jasa aplikasi pemutaran film harus meningkatkan performa sistem rekomendasi untuk kepuasan para pelanggannya ketika menggunakan jasa aplikasi pemutaran film.
### Problem Statement
- Bagaimana cara merekomendasikan film yang disukai dan dapat diminati oleh pengguna dan dijadikan rekomendasi?
### Goal
- Membuat sistem rekomendasi film yang disukai oleh pengguna
### Solution Approach
Solusi yang saya ajukan yaitu dengan menggunakan 2 algoritma machine learning untuk sistem rekomendasi yaitu:
- Collaborative Filtering adalah algoritma yang bergantung pada pendapat komunitas pengguna. Dia tidak memerlukan atribut untuk setiap itemnya. Algoritma ini memberikan rekomendasi berdasarkan nilai rating atau nilai lain, disini saya menggunakan target sebagai dasar penilaian[2].
- Content Based Filtering adalah algoritma yang merekomendasikan item serupa dengan apa yang disukai pengguna, berdasarkan tindakan mereka sebelumnya atau umpan balik eksplisit. Algoritma ini memberikan rekomendasi berdasarkan aktivitas pada masa lalu[3].

## Data Understanding
Data yang digunakan adalah data yang ada pada kaggle [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Dataset memiliki 7 *files* csv tepapi yang kita gunakan untuk sistem rekomendasi adalah:
 - movies_metadata.csv :  File Metadata Film utama. Berisi informasi tentang 45.000 film yang ditampilkan dalam kumpulan data Full MovieLens. Fitur termasuk poster, latar belakang, anggaran, pendapatan, tanggal rilis, bahasa, negara produksi, dan perusahaan.
 - links_small.csv : Berisi ID TMDB (The Movie Database) dan IMDB (Internet Movie Database) dari sebagian kecil dari 9.000 film dari Kumpulan Data Lengkap.
 - ratings_small.csv : Sub kumpulan 100.000 peringkat dari 700 pengguna di 9.000 film.
 
 isi dari movies_metadata.csv
 
 |   | adult | genre    |    id | imdb_id   | original_language | original_title              | release_date | runtime | vote_average | vote_count |
|---|-------|-----------|------:|-----------|-------------------|-----------------------------|--------------|---------|-------------:|------------|
| 1 | false | Animation | 862   | tt0114709 | en                | Toy Story                   |   1995-10-30 | 81.0    | 7.7          | 5415.0     |
| 2 | false | Adventure | 8844  | tt0113497 | en                | Jumanji                     |   1995-12-15 | 104.0   | 6.9          | 2413.0     |
| 3 | false | Romance   | 15602 | tt0113228 | en                | Grumpier Old Men            |   1995-12-22 | 101.0   | 6.5          | 92.0       |
| 4 | false | Comedy    | 31357 | tt0114885 | en                | Waiting to Exhale           |   1995-12-22 | 127.0   | 6.1          | 34.0       |
| 5 | false | Comedy    | 11862 | tt0113041 | en                | Father of the Bride Part II |   1995-02-10 | 106.0   | 5.7          | 173.0      |

isi dari links_small.csv 

|   |  tmbd_id   |
|---|-------|
| 1 |   862 |
| 2 | 8844  |
| 3 | 15602 |
| 4 | 31357 |
| 5 | 11862 |

isi dari ratings_small.csv
|   | userId | movieId | rating |
|---|--------|---------|--------|
| 1 | 1      | 31      | 2.5    |
| 2 | 1      | 1029    | 3.0    |
| 3 | 1      | 1061    | 3.0    |
| 4 | 1      | 1129    | 2.0    |
| 5 | 1      | 1172    | 4.0    |

- adult - merupakan nilai untuk film berkategori berdasarkan umur. ketika nilai adult "false" berarti film tersebut bisa ditonton oleh semua umur, apabila nilai adult "true" maka film tersebut hanya untuk orang dewasa.
- genre - merupakan ragam jenis film
- id - merupakan TMBD id
- imbd_id - merupakan IMBD id
- original_language - merupakan bahasa original film tersebut en berarti english
- original_title - merupakan judul film
- release_date - merupakan tanggal film di rilis
- runtime = durasi film
- vote_average - nilai rata-rata voting
- vote_count - jumlah suara yang memberikan vote
- movieId - TMBD id
- rating - nilai rating

## Data Preparation
Teknik Data Preperation yang digunakan adalah:
-  Menggunakan sebagian dari semua film yang tersedia karena keterbatasan daya komputasi yang tersedia
-  Mengubah jenis data pada "id" menjadi int dan Drop data yang menyebabkan error karena tidak dapat dikonversi
- TrainTestSplit() untuk membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus dilakukan sebelum membuat model. Mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. nilai yang digunakan untuk test adalah 0.25 atau 25% sehingga nilai yang digunakan untuk train adalah 0.75 atau 75% sehingga perbandingan rasio train/test adalah 75:25.

## Modeling
### Content Based Filtering
- Content based filtering menggunakan informasi tentang beberapa item/data untuk merekomendasikan kepada pengguna sebagai referensi mengenai informasi yang digunakan sebelumnya. Tujuan dari content based filtering adalah untuk memprediksi persamaan sejumlah informasi yang didapat dari pengguna. Sebagai contoh, seorang pendengar musik sedang mendengar musik bergenre reggae. Platform musik online secara sistem akan merekomendasikan si pengguna untuk mendengarkan musik lain yang berhubungan dengan reggae. Dalam pembuatannya, content based filtering menggunakan konsep perhitungan Cosine Similarity yang intinya mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama.

- Cosine similarity mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama. Ia menghitung sudut cosinus antara dua vektor. Semakin kecil sudut cosinus, semakin besar nilai cosine similarity.
#### Keuntungan Content Based Filtering
- Model tidak memerlukan data tentang pengguna lain, karena rekomendasi bersifat khusus untuk pengguna ini. Hal ini mempermudah penskalaan ke sejumlah besar pengguna.
- Model ini dapat menangkap minat spesifik pengguna, dan dapat merekomendasikan item khusus yang sangat diminati oleh sedikit pengguna lain
#### Kekurangan Content Based Filtering
- Karena representasi fitur item dirancang secara manual hingga tingkat ini, teknik ini memerlukan banyak pengetahuan domain. Oleh karena itu, model hanya bisa sebaik fitur yang dirancang dengan tangan
- Model hanya dapat membuat rekomendasi berdasarkan minat pengguna yang ada. Dengan kata lain, model memiliki kemampuan terbatas untuk memperluas minat pengguna yang ada
#### 10 TOP-N Recomended Content Base Filtering
Rekomendasi tersebut dihasilkan ketika user menyukai film The Godfather. hasil rekomendasi tersebut didapat berdasarkan genre dari film

|   | movie_name               | genre           |
|---|--------------------------|-----------------|
| 1 | Coming to America (1988) | Comedy\|Romance |
| 2 | Woman of the Year (1942) | Comedy\|Romance |
| 3 | Crossing Delancey (1988) | Comedy\|Romance |
| 4 | Desk Set (1957)          | Comedy\|Romance |
| 5 | Legally Blonde (2001)    | Comedy\|Romance |

### Collorative Filtering
Metode Colaborative filtering merupakan metode yang melakukan proses penyaringan item yang berdasarkan pengguna lain, dengan cara memberikan informasi kepada pengguna berdasarkan kemiripan karakteristik. Dalam pembuatanya saya menggunakan RecommenderNet, pada tahap ini model menghitung skor kecocokan antara pengguna dan musik dengan teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan musik. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan musik. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan musik. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid. Metode ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation.
#### Kelebihan
- Kita tidak memerlukan pengetahuan domain karena penyematan dipelajari secara otomatis.
- Model dapat membantu pengguna menemukan minat baru. Secara terpisah, sistem ML mungkin tidak tahu apakah pengguna tertarik dengan item tertentu, tetapi model mungkin masih merekomendasikannya karena pengguna serupa tertarik pada item tersebut.
- Sampai batas tertentu, sistem hanya memerlukan matriks masukan untuk melatih model faktorisasi matriks. Secara khusus, sistem tidak memerlukan fitur kontekstual. Dalam praktiknya, hal ini dapat digunakan sebagai salah satu dari beberapa generator kandidat.
#### Kekurangan
- Prediksi model untuk pasangan (pengguna, item) tertentu adalah produk titik dari penyematan yang sesuai. Jadi, jika item tidak terlihat selama pelatihan, sistem tidak dapat membuat penyematan untuk item tersebut dan tidak dapat melakukan kueri model dengan item ini. Masalah ini sering disebut masalah cold start
- Fitur samping adalah setiap fitur di luar kueri atau ID item. Untuk rekomendasi film, fitur samping mungkin menyertakan negara atau usia. Menyertakan fitur samping yang tersedia akan meningkatkan kualitas model. Meskipun mungkin tidak mudah untuk menyertakan fitur samping di WALS, generalisasi WALS memungkinkan hal ini.

#### 10 TOP-N Recommended Collaborative Filtering
sadasdasd

## Evaluasi

