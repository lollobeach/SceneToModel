const express = require('express');
const multer = require('multer');
const path = require('path');
   
const app = express();
const port = 3456;

// Configurazione Multer per la gestione degli upload di immagini
const storage = multer.diskStorage({
    destination: './uploads/',
    filename: function(req, file, cb) {
        cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
    }
});

const uploadMiddleware = multer({ storage }).single('image');

app.get('/', (req, res) => {
    res.send('Server Locale Funzionante!');
});

app.post('/upload', (req, res) => {
    // Utilizza il middleware di upload per gestire l'upload dell'immagine
    uploadMiddleware(req, res, function(err) {
        if (err) {
            return res.status(500).send('Errore durante l\'upload dell\'immagine');
        }

        // Controlla se req.file è definito prima di accedere a req.file.filename
        if (req.file) {
            const imageName = req.file.filename;
            const imagePath = `/images/${imageName}`;
            
            // Puoi fare ulteriori operazioni qui, ad esempio salvare il percorso dell'immagine in un database

            // Invia una risposta con il percorso dell'immagine caricata
            res.send(`Immagine ricevuta con successo! Visualizzala qui: <a href="${imagePath}">Visualizza Immagine</a>`);
        } else {
            res.status(400).send('Nessun file ricevuto');
        }
    });
});

app.get('/images/:imageName', (req, res) => {
    const imageName = req.params.imageName;
    res.sendFile(path.join(__dirname, 'uploads', imageName));
});

app.listen(port, () => {
    console.log(`Server in ascolto sulla porta ${port}`);
});
