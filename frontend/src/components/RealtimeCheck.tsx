import React, { useRef, useEffect } from "react";
import CalcAngle from "../functions/ClacAngle";

interface Keypoint {
  x: number;
  y: number;
  score: number;
  name: string;
}
interface Props {
  videoSrc: string;
  keypointsFrames: Keypoint[][];
  fps: number;
  threshold: number;
}

const RealTimeCheck: React.FC<Props> = ({
  videoSrc,
  keypointsFrames,
  fps,
  threshold,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const video = videoRef.current!;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    let rafId: number;

    // 動画メタデータ取得後に canvas サイズを合わせる
    const onLoaded = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    };
    video.addEventListener("loadedmetadata", onLoaded);

    // 描画ループ
    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const frameIdx = Math.floor(video.currentTime * fps);
      const kps = keypointsFrames[frameIdx];
      if (kps) {
        const hip = kps.find((k) => k.name === "Hip");
        const knee = kps.find((k) => k.name === "Knee");
        const ankle = kps.find((k) => k.name === "Ankle");
        if (hip && knee && ankle) {
          const angle = CalcAngle.getAngle(hip, knee, ankle);
          const color = angle < threshold ? "red" : "lime";
          ctx.lineWidth = 4;
          ctx.strokeStyle = color;
          ctx.beginPath();
          ctx.moveTo(hip.x, hip.y);
          ctx.lineTo(knee.x, knee.y);
          ctx.lineTo(ankle.x, ankle.y);
          ctx.stroke();

          // 膝位置にマーカー＆テキスト
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(knee.x, knee.y, 6, 0, Math.PI * 2);
          ctx.fill();
          ctx.font = "16px Arial";
          ctx.fillText(`${angle.toFixed(1)}°`, knee.x + 8, knee.y - 8);
        }
      }
      rafId = requestAnimationFrame(render);
    };

    // 動画再生と同時に描画開始
    video.play().then(() => {
      rafId = requestAnimationFrame(render);
    });

    return () => {
      cancelAnimationFrame(rafId);
      video.removeEventListener("loadedmetadata", onLoaded);
    };
  }, [keypointsFrames, fps, threshold]);

  return (
    <>
      <h1>競歩 フォーム判定 有料モード</h1>
      <h2>動画アップロード</h2>
      <div style={{ position: "relative" }}>
        <video
          ref={videoRef}
          src={videoSrc}
          controls
          style={{ display: "block" }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            pointerEvents: "none",
          }}
        />
      </div>
    </>
  );
};

export default RealTimeCheck;
