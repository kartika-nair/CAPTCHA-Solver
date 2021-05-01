const express = require('express');

const multerMiddleware = require('../middlewares/multerImage.middleware.js');
const errorMiddleware = require('../middlewares/error.middleware.js');
const captchaImageMiddleware = require('../middlewares/captchImage.middleware.js');


const router = express.Router();

//Main
router.post("/image", multerMiddleware.upload.single('captchaImage'), captchaImageMiddleware.postImage)
router.get("/image", captchaImageMiddleware.getAllImages);
router.get("/image/:id", captchaImageMiddleware.getImageByID);
router.delete("/image/:id", captchaImageMiddleware.deleteImageByID);



module.exports = router;
