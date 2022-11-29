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
Data yang digunakan adalah data yang ada pada kaggle [Movie Recommendation Data](https://www.kaggle.com/datasets/rohan4050/movie-recommendation-data)

Dataset memiliki 4 *files* csv yang kita gunakan untuk sistem rekomendasi adalah:
 - movies.csv :  File Metadata Film utama. Berisi informasi tentang 9000 film yang ditampilkan dalam kumpulan data Full MovieLens. Fitur termasuk poster, latar belakang, anggaran, pendapatan, tanggal rilis, bahasa, negara produksi, dan perusahaan.
 - links.csv : Berisi ID TMDB (The Movie Database) dan IMDB (Internet Movie Database) dari sebagian kecil dari 9.000 film dari Kumpulan Data Lengkap.
 - ratings.csv : Sub kumpulan 100.000 peringkat dari 700 pengguna di 9.000 film.
 - tags.csv : berisi label untuk film
 
 isi dari movies.csv
 
|   | movieId | title                              | genres                                          |
|---|---------|------------------------------------|-------------------------------------------------|
| 1 | 1       | Toy Story (1995)                   | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 2 | 2       | Jumanji (1995)                     | Adventure\|Children\|Fantasy                    |
| 3 | 3       | Grumpier Old Men (1995)            | Comedy\|Romance                                 |
| 4 | 4       | Waiting to Exhale (1995)           | Comedy\|Drama\|Romance                          |
| 5 | 5       | Father of the Bride Part II (1995) | Comedy                                          |

isi dari links.csv 

| imdbId  |  tmbdId   | 
|---|-------|
| 114709 |   862 |
| 113497 | 8844  |
| 113228 | 15602 |
| 114885 | 31357 |
| 113041 | 11862 |

isi dari ratings.csv
|   | userId | movieId | rating |
|---|--------|---------|--------|
| 1 | 1      | 1      | 4.0    |
| 2 | 1      | 3    | 4.0    |
| 3 | 1      | 6    | 4.0    |
| 4 | 1      | 47    | 5.0    |
| 5 | 1      | 50    | 5.0    |

isi dari tags.csv

|   | userId | movieID | tag             |
|---|--------|---------|-----------------|
| 1 | 2      | 60756   | funny           |
| 2 | 2      | 60756   | Highly quotable |
| 3 | 2      | 60756   | will ferrell    |
| 4 | 2      | 89774   | Boxing story    |
| 5 | 2      | 89774   | MMA             |

Keterangan kolom:
- genre - merupakan ragam jenis film
- tmbdId - merupakan TMBD id
- imbdId - merupakan IMBD id
- title - merupakan judul film
- movieId - TMBD id
- rating - nilai rating
- tag - label film

## Data Preparation
Teknik Data Preperation yang digunakan adalah:
-  menggabungkan isi dari semua file menjadi satu buah tabel output yang berisi userId,movieId,ratings,timestamp,title,genres.
- menkonversi beberapa data menjadi bentuk list dan dictonary.
- TrainTestSplit() untuk membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus dilakukan sebelum membuat model. Mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. nilai yang digunakan untuk test adalah 0.2 atau 20% sehingga nilai yang digunakan untuk train adalah 0.8 atau 80% sehingga perbandingan rasio train/test adalah 80:20.

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
#### 5 TOP-N Recomended Content Base Filtering
Rekomendasi tersebut dihasilkan ketika user menyukai film Grumpier Old Men (1995). hasil rekomendasi tersebut didapat berdasarkan genre dari film

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

#### 5 TOP-N Recommended Collaborative Filtering
Hasil Filtering tersebut di rekomandasikan untuk user 64 karena user 64 memiliki rating tertinggi dalam genre drama,action

|   | title                                                             | genres                         |
|---|-------------------------------------------------------------------|--------------------------------|
|  1 | Umbrellas of Cherbourg, The (Parapluies de Cherbourg, Les) (1964) | Drama\|Musical\|Romance        |
|  2 | Paths of Glory (1957)                                             | Drama\|War                     |
|  3 | Man Bites Dog (C'est arrivé près de chez vous) (1992)             | Comedy\|Crime\|Drama\|Thriller |
|  4 | Adam's Rib (1949)                                                 | Comedy\|Romance                |
|  5 | Jetée, La (1962)                                                  | Romance\|Sci-Fi                |

## Evaluasi
### Content Based Filtering
|   | movie_name               | genre           |
|---|--------------------------|-----------------|
| 1 | Coming to America (1988) | Comedy\|Romance |
| 2 | Woman of the Year (1942) | Comedy\|Romance |
| 3 | Crossing Delancey (1988) | Comedy\|Romance |
| 4 | Desk Set (1957)          | Comedy\|Romance |
| 5 | Legally Blonde (2001)    | Comedy\|Romance |

Hasil yang didapat menggunakan content based filtering ketika user menyukai film yang berjudul Grumpier Old Men (1995) dengan genre comedy/romance, sistem merekomendasikan 5 film dengan genre yang sama yaitu comedy/romance sehingga nilai presisi dari sistem rekomendasi content based filtering adalah 100%.
### Collaborative Filtering
![image](https://user-images.githubusercontent.com/118952537/204417887-abc30604-ce97-4714-abfc-2899bf421442.png)
- Root Mean Squared Error (RMSE) merupakan salah satu cara untuk mengevaluasi model regresi dengan mengukur tingkat akurasi hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan. Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai observasinya. RMSE menghitung seberapa berbedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati.
## Kesimpulan
 Sistem rekomendasi yang dibuat berdasarkan content based filtering merekomendasikan user berdasarkan genre yang disukai user dan mendapatkan nilai presisi sebesar 100% atau film yang direkomendasi memiliki genre yang sama yang disukai oleh user. Sistem rekomendasi yang dibuat berdasarkan collaborative filtering memiliki nilai RMSE sebesar 0.19 berarti semakin mirip genre yang direkomendasi sistem untuk user.
## Referensi
[1]	“THE EFFECTIVENESS OF WATCHING ENGLISH MOVIE AND THE STUDENTS’ VOCABULARY ENRICHMENT AT THE TENTH GRADE STUDENTS OF MA TERPADU SUWARGI BUWANA DJATI GREGED-CIREBON.”

[2]	B. Schafer, “Collaborative Filtering Recommender Systems DEVELOPMENT OF REMOTE LABORATORIES FOR INTERNET-BASED ENGINEERING EDUCATION View project,” 2007. [Online]. Available: https://www.researchgate.net/publication/200121027

[3]	“Content-Based Recommendation Systems.” [Online]. Available: https://www.researchgate.net/publication/236895069
