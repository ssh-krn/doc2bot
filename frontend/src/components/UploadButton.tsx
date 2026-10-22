interface Props {
  disabled: boolean;
  onFile: (file: File) => Promise<void>;
}

export function UploadButton({ disabled, onFile }: Props) {
  return (
    <label className={`upload${disabled ? " is-disabled" : ""}`}>
      <input
        type="file"
        accept="application/pdf"
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = ""; // allow re-uploading the same file
          if (file) void onFile(file);
        }}
      />
      {disabled ? "Indexing…" : "Upload PDF"}
    </label>
  );
}
