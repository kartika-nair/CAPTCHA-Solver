const CaptchaImage = require('../models/captchaImage.js');
const express = require('express');
const mongoose = require('mongoose');
const multer = require('multer');
require('dotenv').config();
const spawn = require("child_process").spawn;

// Function to spawn a python process
const runScript = (req, res) =>{
    // console.log(process.env.PATH_TO_ML_MODEL, req.file.path)
    return spawn('python', [
        `${process.env.PATH_TO_ML_MODEL}`,
        "--path",
        `./${req.file.path}`,
    ]);
}

// Post Image to server

exports.postImage = (req, res, next) => {
    const captchaImage = new CaptchaImage({
        _id: new mongoose.Types.ObjectId(),
        username: req.body.username,
        captchaImage: req.file.path
    });
    captchaImage
        .save()
        .then(result => {
            const subprocess = runScript(req, res);
            // console.log('AAA');
            var responseFromPythonFile = "";
            var error = ""

            subprocess.stdout.on('data', (data) => {
                responseFromPythonFile = data.toString();
                console.log(data)
                // console.log(responseFromPythonFile)
            });

            subprocess.stderr.on('error', (err) => {
                error = err;
                // console.log(error)

            });
            subprocess.stderr.on('close', () => {
                // console.log('printing test')
                const status = error ? 400 : 201;
                res.status(status).json({
                    createdImage: {
                        username: result.username,
                        _id: result._id,
                    },
                    pythonFileResponse: responseFromPythonFile,
                    error: error
                })
            });
        })
        .catch(err => {
            res.status(400).json({
                error: err,
            })
        })
}

// Get Routes

// Get all images
exports.getAllImages = (req, res, next) => {
  CaptchaImage.find()
    .select("username _id captchaImage")
    .exec()
    .then(docs => {
      const response = {
        count: docs.length,
        images: docs.map(doc => {
          return {
            username: doc.username,
            captchaImage: doc.captchaImage,
            _id: doc._id,
          };
        })
      };
      res.status(200).json(response);
    })
    .catch(err => {
      console.log(err);
      res.status(500).json({
        error: err
      });
    });
};

exports.getImageByID = (req, res, next) => {
    let response, status;
    const id = req.params.id;
    CaptchaImage.find({"_id": id})
        .select("username _id captchaImage")
        .exec()
        .then(doc => {
            if (doc) {
                status = 200;
                response = {
                    captchaImage: doc
                }
            }
            else {
                status = 404;
                response: {
                    message: "No valid entry found"
                }
            }
            res.status(status).json(response)
        })
        .catch (err => {
            res.status(500).json({
                error: err
            });
        });
};


// Delete

exports.deleteImageByID = (req, res, next) => {
    const id = req.params.id;
    CaptchaImage.find({"_id":id})
        .exec()
        .then (result => {
            res.status(200).json({
                message: "Image deleted",
                _id: id
            })
        })
        .catch (err => {
            console.log(err);
            res.status(500).json({
                error: err
            });
        });
};
