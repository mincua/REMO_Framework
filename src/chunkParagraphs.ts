import {encoding_for_model} from "@dqbd/tiktoken";

export const getNoOfTokens = async (input: string): Promise<number> => {
  const encoder = encoding_for_model("gpt-3.5-turbo");
  const tokens = encoder.encode(input);
  encoder.free();

  return tokens.length;
};

// eslint-disable-next-line require-jsdoc
export async function chunkParagraphs(input: {
  text: string
  chunkTokenSize?: number,
}) {
  const {text, chunkTokenSize = 3000} = input;
  const paragraphs = text.split("\n\n");
  const chunkedParagraphs = await chunk(paragraphs, chunkTokenSize);
  console.log("chunkParagraphs: ", chunkedParagraphs.length, " chunks");

  return {
    paragraphs,
    chunks: chunkedParagraphs,
  };
}

// eslint-disable-next-line require-jsdoc
async function chunk(paragraphs: string[], chunkTokenSize: number): Promise<string[]> {
  const chunks: string[] = [];
  let currentChunk: string[] = [];
  let currentChunkTokenCount = 0;
  for (let i = 0; i < paragraphs.length; i++) {
    const paragraph = paragraphs[i];
    const tokenCount = await getNoOfTokens(paragraph);
    if (tokenCount > chunkTokenSize) {
      // If the paragraph is too big to fit in a single chunk,
      // split it into smaller chunks
      const lines = paragraph.split("\n");
      for (let j = 0; j < lines.length; j++) {
        const line = lines[j];
        const lineTokenCount = await getNoOfTokens(line);
        if (currentChunkTokenCount + lineTokenCount <= chunkTokenSize) {
          currentChunk.push(line);
          currentChunkTokenCount += lineTokenCount;
        } else {
          chunks.push(currentChunk.join("\n"));
          currentChunk = [line];
          currentChunkTokenCount = lineTokenCount;
        }
      }
    } else if (currentChunkTokenCount + tokenCount <= chunkTokenSize) {
      // If adding the paragraph to the current chunk
      // will not exceed the maximum size, add it
      currentChunk.push(paragraph);
      currentChunkTokenCount += tokenCount;
    } else {
      // If adding the paragraph to the current chunk
      // will exceed the maximum size, start a new chunk
      chunks.push(currentChunk.join("\n"));
      currentChunk = [paragraph];
      currentChunkTokenCount = tokenCount;
    }
  }
  // Add the final chunk
  chunks.push(currentChunk.join("\n"));
  return chunks;
}
