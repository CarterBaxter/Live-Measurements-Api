import { Platform } from "react-native";

export type MeasurementsResponse = {
  measurements: Record<string, number>;
  debug_info: {
    scale_factor: number | null;
    focal_length: number;
    user_height_cm: number;
  };
};

type UploadOptions = {
  endpointUrl: string;
  frontUri: string;
  leftSideUri?: string;
  heightCm?: number;
  useDepth?: boolean;
  depthMaxDim?: number;
};

const guessMimeType = (uri: string): string => {
  const lower = uri.toLowerCase();
  if (lower.endsWith(".png")) return "image/png";
  if (lower.endsWith(".webp")) return "image/webp";
  return "image/jpeg";
};

const makeFilePart = (uri: string, name: string) => ({
  uri,
  name,
  type: guessMimeType(uri),
});

const makeWebFile = async (uri: string, name: string) => {
  const response = await fetch(uri);
  const blob = await response.blob();
  const type = blob.type || guessMimeType(uri);
  return new File([blob], name, { type });
};

const appendImage = async (
  formData: FormData,
  field: "front" | "left_side",
  uri: string,
  name: string
) => {
  if (Platform.OS === "web") {
    const file = await makeWebFile(uri, name);
    formData.append(field, file);
    return;
  }

  formData.append(field, makeFilePart(uri, name) as any);
};

export const uploadMeasurements = async (
  options: UploadOptions
): Promise<MeasurementsResponse> => {
  const formData = new FormData();

  await appendImage(formData, "front", options.frontUri, "front.jpg");

  if (options.leftSideUri) {
    await appendImage(formData, "left_side", options.leftSideUri, "left_side.jpg");
  }

  if (typeof options.heightCm === "number") {
    formData.append("height_cm", String(options.heightCm));
  }

  formData.append("use_depth", options.useDepth ? "1" : "0");

  if (typeof options.depthMaxDim === "number") {
    formData.append("depth_max_dim", String(options.depthMaxDim));
  }

  const response = await fetch(options.endpointUrl, {
    method: "POST",
    body: formData,
    headers: {
      Accept: "application/json",
    },
  });

  const text = await response.text();
  let payload: unknown = text;
  try {
    payload = text ? JSON.parse(text) : {};
  } catch (error) {
    if (!response.ok) {
      throw new Error(
        `Request failed with ${response.status}: ${response.statusText}`
      );
    }
  }

  if (!response.ok) {
    const errorMessage =
      typeof payload === "object" && payload && "error" in payload
        ? String((payload as { error?: string }).error)
        : `Request failed with ${response.status}: ${response.statusText}`;
    throw new Error(errorMessage);
  }

  return payload as MeasurementsResponse;
};
