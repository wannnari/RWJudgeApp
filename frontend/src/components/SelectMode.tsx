import React from "react";

export type Mode = "upload" | "realTime";

export interface SelectModeProps {
  mode: Mode;
  onChange: (mode: Mode) => void;
}

const SelectMode: React.FC<SelectModeProps> = ({ mode, onChange }) => {
  const modes: { key: Mode; label: string }[] = [
    { key: "upload", label: "簡易モード" },
    { key: "realTime", label: "有料モード" },
  ];

  return (
    <div className="inline-flex rounded-md shadow-sm bg-gray-100">
      {modes.map((m) => (
        <button
          key={m.key}
          onClick={() => onChange(m.key)}
          className={`
                    px-4 py-2 first:rounded-l-md last:rounded-r-md
                    focus:outline-none
                    ${
                      mode === m.key
                        ? "bg-white text-blue-600 font-semibold shadow"
                        : "text-gray-600 hover:bg-white"
                    }
                `}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
};

export default SelectMode;
