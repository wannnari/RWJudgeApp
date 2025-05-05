import React, { useState } from "react";
import axios from "axios";
import Loading from "./loading";

const UploadForm = () => {
  const [file, setFile] = useState<File>();
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [isLoading, setLoading] = useState<boolean>(false);
  const [message, setMessage] = useState<String>("");
  const [isError, setError] = useState<boolean>(false);

  // ファイルが選択された時
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  // アップロードボタンを押したとき
  const handleUpload = async () => {
    if (!file) {
      setError(true);
      return;
    }
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://localhost:8000/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      console.log("サーバからのレスポンス:", response.data);
      if (response.data.image_base64) {
        setImageBase64(response.data.image_base64);
        setMessage(response.data.message);
        setError(false);
      }
    } catch (error) {
      console.error("アップロードエラー:", error);
    }
    setLoading(false);
  };

  return (
    <div>
      {isLoading && <Loading />}
      <h1>競歩 フォーム判定</h1>
      <h2>動画アップロード</h2>
      <input type="file" accept="video/*" onChange={handleFileChange} />
      <button onClick={handleUpload}>アップロード</button>
      {imageBase64 && (
        <div>
          <img
            src={`data:image/jpeg;base64, ${imageBase64}`}
            alt="膝が曲がっていたフレーム"
            style={{ width: "300px", marginTop: "20px" }}
          />
          <div>{message}</div>
        </div>
      )}
      {isError && <div>動画ファイルをセットしてください</div>}
    </div>
  );
};

export default UploadForm;
