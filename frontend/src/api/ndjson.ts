/**
 * Reads a newline-delimited JSON stream and calls onItem for each parsed object.
 * Handles chunks that split a line across reads, and a final line without a trailing newline.
 */
export async function readNdjson<T>(
  body: ReadableStream<Uint8Array>,
  onItem: (item: T) => void,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const emit = (line: string) => {
    const trimmed = line.trim();
    if (trimmed) onItem(JSON.parse(trimmed) as T);
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline = buffer.indexOf("\n");
    while (newline >= 0) {
      emit(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
      newline = buffer.indexOf("\n");
    }
  }

  buffer += decoder.decode();
  emit(buffer);
}
