const express = require('express');
const axios = require('axios');
const mysql = require('mysql2/promise');

const app = express();
const port = 3000;

app.use(express.json());

// Konfigurasi koneksi ke Google Cloud SQL
const dbConfig = {
  host: 'your-google-cloud-sql-ip',
  user: 'your-database-user',
  password: 'your-database-password',
  database: 'your-database-name',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
};

// Endpoint untuk menyimpan data warna urine ke database
app.post('/save_urine_color', async (req, res) => {
  try {
    const { color } = req.body;

    // Buat koneksi ke database
    const connection = await mysql.createConnection(dbConfig);

    // Simpan data warna urine ke database
    const [result] = await connection.execute('INSERT INTO urine_colors (color) VALUES (?)', [color]);

    // Tutup koneksi
    await connection.end();

    res.json({ status: 'success', message: 'Data saved to database.' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ status: 'error', message: 'Internal Server Error' });
  }
});

// Endpoint untuk mengambil data warna urine dari database
app.get('/get_urine_colors', async (req, res) => {
  try {
    // Buat koneksi ke database
    const connection = await mysql.createConnection(dbConfig);

    // Ambil semua data warna urine dari database
    const [rows] = await connection.execute('SELECT * FROM urine_colors');

    // Tutup koneksi
    await connection.end();

    res.json({ status: 'success', data: rows });
  } catch (error) {
    console.error(error);
    res.status(500).json({ status: 'error', message: 'Internal Server Error' });
  }
});

// Endpoint untuk mengonsumsi API deteksi warna urine
app.post('/detect_urine_color_mobile', async (req, res) => {
  try {
    // Ambil data gambar dari payload yang diterima
    const imageData = req.body.imageData;

    // Kirim permintaan ke endpoint API deteksi warna urine
    const response = await axios.post('http://34.128.118.210:5000/predict', {
      imageData: imageData,
    });

    // Simpan data warna urine ke database
    const { color } = response.data;

    // Buat koneksi ke database
    const connection = await mysql.createConnection(dbConfig);

    // Simpan data warna urine ke database
    const [result] = await connection.execute('INSERT INTO urine_colors (color) VALUES (?)', [color]);

    // Tutup koneksi
    await connection.end();

    // Kirim kembali respons dari API deteksi warna urine ke pengguna mobile
    res.json(response.data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ status: 'error', message: 'Internal Server Error' });
  }
});

// Jalankan server
app.listen(port, () => {
  console.log(`Server berjalan di http://localhost:${port}`);
});
