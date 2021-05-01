import { base_url } from '../config.json';
import axios from 'axios';


export const uploadImage = async (file, username) => {
    let formData = new FormData();

    formData.append("captchaImage", file);
    formData.append("username", username);

    const response = await axios({
        url: `${base_url}/captcha/image`,
        method: "POST",
        headers: {
            'Content-Type': 'multipart/form-data'
        },
        data: formData
    })
    return response.data;
}
