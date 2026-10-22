/** Element id for a source card. Scoped by message so [n] links from one answer don't hit another's sources. */
export const sourceAnchorId = (messageId: string, index: number) => `src-${messageId}-${index}`;
