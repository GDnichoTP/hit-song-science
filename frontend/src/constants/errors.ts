// constants/errors.ts
export type ErrorCode =
  | "UNSUPPORTED_FORMAT"
  | "CORRUPT_FILE"
  | "FILE_TOO_LARGE"
  | "DURATION_TOO_LONG"
  | "DURATION_TOO_SHORT"
  | "SILENT_AUDIO"
  | "UNPROCESSABLE_AUDIO"
  | "FEATURE_SERVICE_UNAVAILABLE"
  | "FEATURE_SERVICE_BUSY"
  | "FEATURE_SERVICE_FAILED"
  | "NETWORK_ERROR"
  | "UNKNOWN_ERROR";

export interface ErrorInfo {
  title: string; // short heading, e.g. "Corrupt file"
  message: string; // what went wrong
  action: string; // what the user should do next
}

// Validation limits: one place for the numbers.
// Must match backend/utils/validation.py (MAX_SIZE, MIN/MAX_UPLOAD_DURATION).
export const LIMITS = {
  formats: ["WAV", "MP3"],
  maxSizeMb: 15,
  minDurationS: 10,
  maxDurationS: 30,
} as const;

// English number format: 0.5 -> "0.5"
export const fmt = (n: number) => n.toLocaleString("en-US");

export const ERRORS: Record<ErrorCode, ErrorInfo> = {
  UNSUPPORTED_FORMAT: {
    title: "Unsupported format",
    message: `The file must be ${LIMITS.formats.join(" or ")}.`,
    action: `Convert the file to ${LIMITS.formats.join(" or ")}, then upload it again.`,
  },
  CORRUPT_FILE: {
    title: "Corrupt file",
    message: "The file could not be read as audio.",
    action: "Make sure the file plays, then upload it again.",
  },
  FILE_TOO_LARGE: {
    title: "File too large",
    message: `The file is larger than the ${fmt(LIMITS.maxSizeMb)} MB limit.`,
    action: "Save the clip as MP3, then upload it again.",
  },
  DURATION_TOO_LONG: {
    title: "Clip too long",
    message: `The clip is longer than the ${fmt(LIMITS.maxDurationS)} second limit.`,
    action: `Cut ${fmt(LIMITS.maxDurationS)} seconds of the song, for example the chorus, then upload it again.`,
  },
  DURATION_TOO_SHORT: {
    title: "Clip too short",
    message: `The clip is shorter than ${fmt(LIMITS.minDurationS)} seconds.`,
    action: `Upload ${fmt(LIMITS.minDurationS)} to ${fmt(LIMITS.maxDurationS)} seconds of the song.`,
  },
  SILENT_AUDIO: {
    title: "Only silence",
    message: "No sound was detected in the file.",
    action: "Upload a part of the song where the music can be heard.",
  },
  UNPROCESSABLE_AUDIO: {
    title: "Audio could not be analyzed",
    message: "Audio features could not be computed from this file.",
    action: "Upload another part of the song.",
  },
  FEATURE_SERVICE_UNAVAILABLE: {
    title: "Audio analysis unavailable",
    message: "The audio analysis service (ReccoBeats) did not respond.",
    action: "Try again in a few minutes.",
  },
  FEATURE_SERVICE_BUSY: {
    title: "Too many requests",
    message: "The audio analysis service (ReccoBeats) is limiting requests right now.",
    action: "Wait a minute, then try again.",
  },
  FEATURE_SERVICE_FAILED: {
    title: "Audio analysis failed",
    message: "The audio analysis service (ReccoBeats) could not analyze this clip.",
    action: "Upload another part of the song.",
  },
  NETWORK_ERROR: {
    title: "Could not connect",
    message: "The request did not reach the server.",
    action: "Check your internet connection, then try again.",
  },
  UNKNOWN_ERROR: {
    title: "Something went wrong",
    message: "The file could not be processed.",
    action: "Try again in a moment.",
  },
};

export function isErrorCode(value: unknown): value is ErrorCode {
  return typeof value === "string" && value in ERRORS;
}

// Always returns something displayable, even for unexpected codes
export function getError(code: unknown): ErrorInfo {
  return isErrorCode(code) ? ERRORS[code] : ERRORS.UNKNOWN_ERROR;
}
