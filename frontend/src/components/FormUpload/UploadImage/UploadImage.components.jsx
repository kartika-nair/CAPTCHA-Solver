import React, {useState} from 'react';
import classes from './Uploadimage.module.css';

import {uploadImage} from '../../../helpers/image.js';
import Answer from '../../Answer/Answer.components.jsx';


const UploadImage = () => {
    const [username, setUsername] = useState();
    const [file, setFile] = useState();
    const [answer, setAnswer] = useState();
    const [loading, setLoading] = useState("null");

    const send = async () => {
        setLoading("spinner");
        const response = await uploadImage(file, username);
        setAnswer(response["pythonFileResponse"]);
        setLoading("answer");
    };

    return (
            <>
            <div className = "UploadForm">
                <form action = "#">
                    <div className = {classes.flex}>
                        <label htmlFor="name">Username</label>
                        <input
                            type = "text"
                            id = "name"
                            onChange = {(event) => setUsername(event.target.value)}
                        />
                        <label htmlFor="name">Image</label>
                        <input
                            type = "file"
                            id = "file"
                            onChange = {(event) => setFile(event.target.files[0])}
                        />
                    </div>
                </form>
                <button onClick={() => send()}>Submit</button>
            </div>
            <Answer
                loading = {loading}
                answer = {answer}
            />
            </>
    )
}

export default UploadImage;
