// Regression test for the Gold <-> Silver unit-mismatch 422 loop.
// The original bug: one shared "unit" state persisted across metal switches
// (e.g. "10gram" carried into silver, "kg" into gold), producing 422s from the
// API and a "Failed to load data / Retry" loop that never self-healed.
// These tests pin the single source of truth (UNITS_BY_METAL) and the
// sanitize/reset behavior so a future metal cannot reintroduce the bug.
import { describe, expect, it } from "vitest";
import {
  KARATS_BY_GOLD,
  QUALITY_DEFAULTS_BY_METAL,
  UNITS_BY_METAL,
  isValidUnit,
  sanitizeQualityParams,
  sanitizeUnit,
} from "./metalUnits";

describe("metal switch keeps unit valid (regression: unit-mismatch 422 loop)", () => {
  it("Gold -> Silver -> Gold: the unit is valid for the selected metal at every step", () => {
    let metal: "gold" | "silver" = "gold";
    let unit = "10gram";
    expect(isValidUnit(metal, unit)).toBe(true);

    // user clicks Silver: handler sets the new metal's own default atomically
    metal = "silver";
    unit = QUALITY_DEFAULTS_BY_METAL[metal].unit;
    expect(isValidUnit(metal, unit)).toBe(true);
    expect(unit).toBe("kg");

    // user clicks back to Gold: default resets again
    metal = "gold";
    unit = QUALITY_DEFAULTS_BY_METAL[metal].unit;
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
      expect(
        units.some((u) => u.id === QUALITY_DEFAULTS_BY_METAL[metal].unit)
      ).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// Regression: "karat applies to gold only — use fineness for silver" 422.
// The previous fix only covered the metal SWITCH; this bug appeared when the
// user changed a DIFFERENT control (horizon) while already on Silver, because
// the karat state from a previous Gold session was still being sent.
// These tests sanitize quality params for EVERY control change, per metal.
// ---------------------------------------------------------------------------
describe("quality params (karat/fineness) stay valid for the current metal", () => {
  it("changing horizon/range/granularity/currency while on Silver never sends a gold karat", () => {
    // User had Gold+22K earlier; the karat state is stale. Now on Silver.
    const stale = { unit: "kg", karat: "22k", fineness: "999" };
    // Each control change re-runs load() -> sanitize; karat must be forced
    // to silver's default and never sent as a gold karat.
    const q = sanitizeQualityParams("silver", stale);
    expect(q.karat).toBe("24k");
    expect(q.unit).toBe("kg");
    expect(q.fineness).toBe("999");
  });

  it("changing controls while on Gold never sends a silver fineness", () => {
    const stale = { unit: "10gram", karat: "22k", fineness: "925" };
    const q = sanitizeQualityParams("gold", stale);
    expect(q.fineness).toBe("999");
    expect(q.karat).toBe("22k"); // gold karat preserved
    expect(q.unit).toBe("10gram");
  });

  it("stale fineness from a silver session is reset when switching to gold", () => {
    const q = sanitizeQualityParams("gold", { unit: "10gram", karat: "24k", fineness: "958" });
    expect(q.fineness).toBe("999");
  });

  it("invalid fineness on silver falls back to 999 default", () => {
    const q = sanitizeQualityParams("silver", { unit: "kg", karat: "24k", fineness: "22k" });
    expect(q.fineness).toBe("999");
  });

  it("metal switch + immediate control change in one interaction stays valid", () => {
    // Real-user click pattern: switch to Silver AND change horizon before the
    // next fetch. handleMetalChange resets quality defaults atomically; the
    // subsequent load() re-sanitizes regardless of what state is present.
    const defaults = QUALITY_DEFAULTS_BY_METAL["silver"];
    const afterSwitch = sanitizeQualityParams("silver", {
      unit: defaults.unit,
      karat: "22k", // stale gold karat, still present mid-transition
      fineness: "999",
    });
    expect(afterSwitch.karat).toBe("24k");
    expect(afterSwitch.unit).toBe("kg");
    expect(afterSwitch.fineness).toBe("999");
  });

  it("every karat in the gold selector and fineness in the silver selector is accepted", () => {
    for (const k of KARATS_BY_GOLD) {
      const q = sanitizeQualityParams("gold", { unit: "10gram", karat: k, fineness: "999" });
      expect(q.karat).toBe(k);
    }
    for (const f of ["999", "958", "925"] as const) {
      const q = sanitizeQualityParams("silver", { unit: "kg", karat: "24k", fineness: f });
      expect(q.fineness).toBe(f);
    }
  });
});
