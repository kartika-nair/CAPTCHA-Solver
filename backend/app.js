const express = require('express');
const cookieParser = require('cookie-parser');
require('dotenv').config();
const mongoose = require('mongoose');
const cors = require('cors');

const app = express();

//Import Paths
const captchaImage = require('./routes/captchaimage.route.js');

// Middlewares
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({
    extended: false
}));
app.use(cookieParser());
app.use('/uploads', express.static('uploads'));


// Connect to DB
mongoose.connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
    useFindAndModify: false,
    useCreateIndex: true,
}).then(() => {
    console.log(`Connected to DB`);
}).catch(err => {
    console.log(`[CONNECTING TO DB] Error: ${err}`)
});


//Paths and routes
app.use('/captcha', captchaImage);

// Starting the server
const port = process.env.PORT || 5000;
app.listen(port, () => {
    console.log(`[APP START] Server listening on PORT: ${port}`);
});

