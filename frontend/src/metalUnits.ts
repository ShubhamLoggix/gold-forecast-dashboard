// Single source of truth for valid units per metal.
// Root-cause fix for the 422 unit-mismatch bug on Gold <-> Silver switches:
// unit state must always be validated against the CURRENT metal before any
// API call, and metal switches reset the unit atomically in the event handler
// (never via a post-render effect that races the fetch).
import type { Metal, Unit } from "./api";

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
