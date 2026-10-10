import type { SourceImage } from "../types";

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Could not read the selected image."));
    reader.readAsDataURL(file);
  });
}

function readDimensions(dataUrl: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => reject(new Error("The selected file is not a readable image."));
    image.src = dataUrl;
  });
}

export async function createSourceImage(file: File): Promise<SourceImage> {
  if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
    throw new Error("Choose a PNG, JPEG, or WebP image.");
  }
  if (file.size > 24 * 1024 * 1024) {
    throw new Error("Source images must be smaller than 24 MB.");
  }
  const dataUrl = await readAsDataUrl(file);
  const dimensions = await readDimensions(dataUrl);
  if (dimensions.width * dimensions.height > 24_000_000) throw new Error("Source images must be at most 24 megapixels.");
  return {
    name: file.name,
    dataUrl,
    base64: dataUrl.slice(dataUrl.indexOf(",") + 1),
    ...dimensions,
  };
}

export function downloadDataUrl(dataUrl: string, filename: string): void {
  const anchor = document.createElement("a");
  anchor.href = dataUrl;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
}

export async function copyDataUrl(dataUrl: string): Promise<void> {
  const blob = await (await fetch(dataUrl)).blob();
  if (!navigator.clipboard?.write || typeof ClipboardItem === "undefined") {
    throw new Error("Image copying is not supported by this browser. Use Download PNG instead.");
  }
  const pngBlob = blob.type === "image/png"
    ? blob
    : await new Promise<Blob>((resolve, reject) => {
        const image = new Image();
        image.onload = () => {
          const canvas = document.createElement("canvas");
          canvas.width = image.naturalWidth;
          canvas.height = image.naturalHeight;
          canvas.getContext("2d")?.drawImage(image, 0, 0);
          canvas.toBlob((value) => value ? resolve(value) : reject(new Error("Could not copy image.")), "image/png");
        };
        image.onerror = () => reject(new Error("Could not copy image."));
        image.src = dataUrl;
      });
  await navigator.clipboard.write([new ClipboardItem({ "image/png": pngBlob })]);
}
