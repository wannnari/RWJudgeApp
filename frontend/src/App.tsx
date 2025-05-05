import React, { useState } from "react";
import "./App.css";
import SelectMode, { Mode } from "./components/SelectMode";
import RealTimeCheck from "./components/RealtimeCheck";
import UploadForm from "./components/UploadForm";

const App: React.FC = () => {
  const [selectMode, setSelectedMode] = useState<Mode>("upload");

  return (
    <div className="App">
      <SelectMode mode={selectMode} onChange={setSelectedMode} />
      {selectMode === "upload" && <UploadForm />}
      {selectMode === "realTime" && <RealTimeCheck />}
    </div>
  );
};

export default App;
