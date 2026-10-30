// Must match SUPPORTED_EXTENSIONS in backend/app/parsing/__init__.py.
// The server stays the authority; this list only gives instant feedback before upload.

export const FILE_GROUPS: Record<string, string[]> = {
  PDF: [".pdf"],
  Word: [".doc", ".docx"],
  PowerPoint: [".ppt", ".pptx"],
  Excel: [".xls", ".xlsx", ".xlsm"],
  Image: [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"],
};

const ALL_EXTENSIONS = Object.values(FILE_GROUPS).flat();

export const ACCEPT_ATTRIBUTE = ALL_EXTENSIONS.join(",");

export function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot >= 0 ? filename.slice(dot).toLowerCase() : "";
}

export function isSupportedFile(filename: string): boolean {
  return ALL_EXTENSIONS.includes(extensionOf(filename));
}

export function kindLabel(kind: string): string {
  const ext = `.${kind.toLowerCase()}`;
  const group = Object.entries(FILE_GROUPS).find(([, exts]) => exts.includes(ext));
  return group ? group[0] : kind.toUpperCase();
}

export const MODALITY_LABEL: Record<string, string> = {
  text: "Text",
  table: "Table",
  image: "Image",
  equation: "Equation",
};
