var mongoose = require('mongoose');

var captchaImageSchema = new mongoose.Schema({
    _id : mongoose.Schema.Types.ObjectId,
    username: {
        type: String,
        required: true
    },
    captchaImage:
    {
        type: String,
        required: true
    }
});

module.exports = new mongoose.model('CaptchaImage', captchaImageSchema);
