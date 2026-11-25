// constants/labels.ts
// Ids must match LABEL_IDS in backend/services/predictor.py
export type ClassId = "popular" | "not_popular";

export interface ClassInfo {
  id: ClassId;
  name: string; // display name
  description: string; // short English description
}

export const CLASS_BY_ID: Record<ClassId, ClassInfo> = {
  popular: {
    id: "popular",
    name: "Popular",
    description: "Likely to chart within 8 weeks",
  },
  not_popular: {
    id: "not_popular",
    name: "Not popular",
    description: "Unlikely to chart within 8 weeks",
  },
};

export const CLASS_ORDER: readonly ClassId[] = ["popular", "not_popular"];
export const CLASSES: readonly ClassInfo[] = CLASS_ORDER.map(
  (id) => CLASS_BY_ID[id],
);

export function isClassId(value: unknown): value is ClassId {
  return typeof value === "string" && value in CLASS_BY_ID;
}
