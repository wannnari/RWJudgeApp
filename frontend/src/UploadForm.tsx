import React,{ useState } from 'react';
import axios from 'axios';

const UploadForm = () => {
    const [file, setFile] = useState<File>();
    const [imageBase64, setImageBase64] = useState<string | null>(null);

    // ファイルが選択された時
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if(selectedFile){
        setFile(selectedFile);
        }
    }

    // アップロードボタンを押したとき
    const handleUpload = async() => {
    if(!file){
        alert('ファイルを選択してください');
        return;
    }
    const formData = new FormData();
    formData.append('file', file);

    try{
        const response = await axios.post('http://localhost:8000/upload', formData, {
            headers: {
                'Content-Type' : 'multipart/form-data',
            },
            });
            console.log('サーバからのレスポンス:', response.data);
            if (response.data.image_base64) {
                setImageBase64(response.data.image_base64);
            }
        }catch(error){
            console.error('アップロードエラー:', error);
        }
    };
    
    return (
        <div>
            <h2>動画アップロード</h2>
            <input type="file" accept="video/*" onChange={handleFileChange} />
            <button onClick={handleUpload}>アップロード</button>
            {
                imageBase64 && (
                    <img
                    src={`data:image/jpeg;base64, ${imageBase64}`}
                    alt="膝が曲がっていたフレーム"
                    style={{width: '300px', marginTop: '20px'}}
                    />
                )
            }
        </div>
    );
};

export default UploadForm;