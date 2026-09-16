// Single source of truth for valid units per metal.
// Root-cause fix for the 422 unit-mismatch bug on Gold <-> Silver switches:
// unit state must always be validated against the CURRENT metal before any
// API call, and metal switches reset the unit atomically in the event handler
// (never via a post-render effect that races the fetch).
import type { Fineness, Karat, Metal, Unit } from "./api";

export const UNITS_BY_METAL: Record<Metal, { id: Unit; label: string }[]> = {
  gold: [
    { id: "10gram", label: "per 10g" },
    { id: "gram", label: "per g" },
  ],
  silver: [
    { id: "kg", label: "per kg" },
    { id: "gram", label: "per g" },
  ],
};

export const DEFAULT_UNIT_BY_METAL: Record<Metal, Unit> = {
  gold: "10gram",
  silver: "kg",
};

export function sanitizeUnit(
  metal: Metal,
  unit: Unit | string | undefined | null,
): Unit {
  const found = UNITS_BY_METAL[metal].find((u) => u.id === unit);
  return found ? found.id : DEFAULT_UNIT_BY_METAL[metal];
}

export function isValidUnit(metal: Metal, unit: Unit | string | undefined | null): boolean {
  return UNITS_BY_METAL[metal].some((u) => u.id === unit);
}

// ---------------------------------------------------------------------------
// Quality params (karat / fineness) — same bug class as unit, generalized.
// The backend requires the NON-applicable field to keep its default value:
// gold rejects fineness != "999"; silver rejects karat != "24k".
// ---------------------------------------------------------------------------
export interface QualityParams {
  unit: Unit;
  karat: Karat;
  fineness: Fineness;
}

export const KARATS_BY_GOLD: Karat[] = ["24k", "22k", "18k"];
export const FINENESSES_BY_SILVER: Fineness[] = ["999", "958", "925"];

export const QUALITY_DEFAULTS_BY_METAL: Record<Metal, QualityParams> = {
  gold: { unit: "10gram", karat: "24k", fineness: "999" },
  silver: { unit: "kg", karat: "24k", fineness: "999" },
};

export function sanitizeQualityParams(
  metal: Metal,
  params: { unit?: Unit | string | null; karat?: string | null; fineness?: string | null },
): QualityParams {
  const defaults = QUALITY_DEFAULTS_BY_METAL[metal];
  if (metal === "silver") {
    return {
      unit: isValidUnit("silver", params.unit) ? (params.unit as Unit) : defaults.unit,
      karat: "24k", // gold-only field forced to default for silver
      fineness: FINENESSES_BY_SILVER.includes(params.fineness as Fineness)
        ? (params.fineness as Fineness)
        : defaults.fineness,
    };
  }
  return {
    unit: isValidUnit("gold", params.unit) ? (params.unit as Unit) : defaults.unit,
    karat: KARATS_BY_GOLD.includes(params.karat as Karat)
      ? (params.karat as Karat)
      : defaults.karat,
    fineness: "999", // silver-only field forced to default for gold
  };
}
