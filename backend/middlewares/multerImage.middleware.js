var multer = require('multer');
require('dotenv').config();

var storage;

const fileFilter = (req, file, cb) => {
  // reject a file
  if (file.mimetype === 'image/jpeg' || file.mimetype === 'image/png' || file.mimetype === 'image/jpg') {
    cb(null, true);
  } else {
    cb(null, false);
  }
};

storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, "./uploads/");
    },
    filename: (req, file, cb) => {
        cb(
            null,
            file.fieldname + "-" + Date.now() + "_" + file.originalname
        );
    },
});

exports.upload = multer({ 
    storage: storage,
    limits: {
        fileSize: 1024 * 1024 * 5 //file size capped to 5mb
    }, 
    fileFilter: fileFilter
});
