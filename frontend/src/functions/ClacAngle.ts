const CalcAngle = () => {
  // 2点間ベクトル
  const vec = (a: Keypoint, b: Keypoint) => [b.x - a.x, b.y - a.y];
  // 内積
  const dot = (u: number[], v: number[]) => u[0] * v[0] + u[1] * v[1];
  // ノルム
  const norm = (u: number[]) => Math.hypot(u[0], u[1]);

  /**
   * 3点 a–b–c の ∠abc を度数法で返す
   */

  const getAngle = (a: Keypoint, b: Keypoint, c: Keypoint): number => {
    const v1 = vec(b, a);
    const v2 = vec(b, c);
    const cosθ = dot(v1, v2) / (norm(v1) * norm(v2) + 1e-6);
    return (Math.acos(Math.min(Math.max(cosθ, -1), 1)) * 180) / Math.PI;
  };
};

export default CalcAngle;
