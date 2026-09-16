// Regression test for the Gold <-> Silver unit-mismatch 422 loop.
// The original bug: one shared "unit" state persisted across metal switches
// (e.g. "10gram" carried into silver, "kg" into gold), producing 422s from the
// API and a "Failed to load data / Retry" loop that never self-healed.
// These tests pin the single source of truth (UNITS_BY_METAL) and the
// sanitize/reset behavior so a future metal cannot reintroduce the bug.
import { describe, expect, it } from "vitest";
import {
  DEFAULT_UNIT_BY_METAL,
  UNITS_BY_METAL,
  isValidUnit,
  sanitizeUnit,
} from "./metalUnits";

describe("metal switch keeps unit valid (regression: unit-mismatch 422 loop)", () => {
  it("Gold -> Silver -> Gold: the unit is valid for the selected metal at every step", () => {
    let metal: "gold" | "silver" = "gold";
    let unit = "10gram";
    expect(isValidUnit(metal, unit)).toBe(true);

    // user clicks Silver: handler sets the new metal's own default atomically
    metal = "silver";
    unit = DEFAULT_UNIT_BY_METAL[metal];
    expect(isValidUnit(metal, unit)).toBe(true);
    expect(unit).toBe("kg");

    // user clicks back to Gold: default resets again
    metal = "gold";
    unit = DEFAULT_UNIT_BY_METAL[metal];
    expect(isValidUnit(metal, unit)).toBe(true);
    expect(unit).toBe("10gram");
  });

  it("the previously-selected silver unit never leaks into gold requests", () => {
    expect(isValidUnit("silver", "kg")).toBe(true);
    expect(isValidUnit("gold", "kg")).toBe(false);
    // sanitize replaces the invalid carry-over instead of sending it
    expect(sanitizeUnit("gold", "kg")).toBe("10gram");
  });

  it("the other direction: gold's 10gram never leaks into silver requests", () => {
    expect(isValidUnit("gold", "10gram")).toBe(true);
    expect(isValidUnit("silver", "10gram")).toBe(false);
    expect(sanitizeUnit("silver", "10gram")).toBe("kg");
  });

  it("gram is valid for both metals and survives switches unchanged", () => {
    expect(isValidUnit("gold", "gram")).toBe(true);
    expect(isValidUnit("silver", "gram")).toBe(true);
    expect(sanitizeUnit("silver", "gram")).toBe("gram");
    expect(sanitizeUnit("gold", "gram")).toBe("gram");
  });

  it("sanitize falls back to the metal default for any unknown/undefined unit", () => {
    expect(sanitizeUnit("gold", undefined)).toBe("10gram");
    expect(sanitizeUnit("silver", "cups" as unknown as string)).toBe("kg");
  });

  it("each metal declares a non-empty unit list containing its default", () => {
    for (const metal of ["gold", "silver"] as const) {
      const units = UNITS_BY_METAL[metal];
      expect(units.length).toBeGreaterThan(0);
      expect(units.some((u) => u.id === DEFAULT_UNIT_BY_METAL[metal])).toBe(true);
    }
  });
});
