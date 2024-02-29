const express = require('express');
const multer = require('multer');
const path = require('path');

const app = express();
const port = 3000;

// Configurazione Multer per la gestione degli upload di immagini
const storage = multer.diskStorage({
    destination: './uploads/',
    filename: function(req, file, cb) {
        cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
    }
});

const upload = multer({ storage });

app.get('/', (req, res) => {
    res.send('Server Locale Funzionante!');
});

app.post('/upload', upload.single('image'), (req, res) => {
    // Gestisci l'upload dell'immagine e invia una risposta appropriata
    res.send('Immagine ricevuta con successo!');
});

app.get('/images/:imageName', (req, res) => {
    const imageName = req.params.imageName;
    res.sendFile(path.join(__dirname, 'uploads', imageName));
});

app.listen(port, () => {
    console.log(`Server in ascolto sulla porta ${port}`);
});