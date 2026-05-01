//! `DatetimeParseTool` — convert natural-language date/time strings
//! into ISO 8601 normalized output.
//!
//! # Why a dedicated tool
//!
//! Agents constantly receive user-provided dates in unstructured form:
//! "show me errors from last Tuesday", "fetch metrics for the past
//! 3 days", "what was the close price on March 15?". Without a
//! parser, the agent has to do date math on `CurrentTimeTool`'s
//! output (iter 279), which is brittle and fails for anything beyond
//! "today / yesterday." This tool turns natural-language strings
//! into ISO 8601 + a structured breakdown the agent can feed
//! directly into SQL queries, log filters, or API parameters.
//!
//! # What this is NOT
//!
//! This is NOT a full natural-language date library. It handles the
//! 80% of agent-facing patterns:
//!
//! - `today` / `yesterday` / `tomorrow` / `now`
//! - `N hours ago`, `N days ago`, `N weeks ago`, `N months ago`
//! - `in N hours/days/weeks/months`
//! - `last <weekday>` / `next <weekday>`
//! - ISO 8601 date / datetime (passed through, validated)
//! - `YYYY-MM-DD` (passed through with reference-tz time-of-day = 0)
//!
//! Things that are intentionally NOT supported:
//!
//! - Quarter literals (`Q1 2025`) — too ambiguous; agents should
//!   reformulate as date ranges.
//! - Timezone-name parsing (`5pm PST`) — chrono's `chrono-tz` would
//!   pull in the IANA database; this tool stays small.
//! - Holiday names (`Christmas`, `Easter`) — would need a calendar.
//! - Fuzzy/typo tolerance — the agent should retry with a corrected
//!   string.
//!
//! For unsupported strings, the tool errors with `Error::InvalidInput`
//! so the LLM can see what failed and retry with a different wording.

use async_trait::async_trait;
use chrono::{
    DateTime, Datelike, Duration, FixedOffset, NaiveDate, NaiveDateTime, NaiveTime, TimeZone,
    Timelike, Utc, Weekday,
};
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, Default)]
pub struct DatetimeParseTool;

impl DatetimeParseTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for DatetimeParseTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "datetime_parse".into(),
            description: "Parse a natural-language date/time string into ISO 8601. Handles: \
                'today' / 'yesterday' / 'tomorrow' / 'now'; relative offsets like '3 days ago', \
                '2 weeks ago', 'in 5 hours'; weekday phrases 'last Monday' / 'next Friday'; \
                ISO 8601 datetimes (passed through); 'YYYY-MM-DD' date-only (time set to 00:00). \
                Returns {iso8601, date, weekday, unix}. Reference time defaults to current UTC; \
                override with `reference` (ISO 8601). Set `tz_offset_hours` to interpret the \
                input in a non-UTC timezone (e.g. -5 for US/Eastern, 5.5 for India)."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The natural-language date/time string to parse."
                    },
                    "reference": {
                        "type": "string",
                        "description": "Optional ISO 8601 reference timestamp. Relative phrases \
                            ('3 days ago', 'next Friday') are computed relative to this. Defaults to current UTC."
                    },
                    "tz_offset_hours": {
                        "type": "number",
                        "description": "Optional timezone offset in hours (fractional ok: 5.5 for India). Defaults to 0 (UTC)."
                    }
                },
                "required": ["input"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let input = args
            .get("input")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("datetime_parse: missing `input`"))?;
        let tz_offset_hours = args
            .get("tz_offset_hours")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let offset_secs = (tz_offset_hours * 3600.0) as i32;
        let offset = FixedOffset::east_opt(offset_secs).ok_or_else(|| {
            Error::invalid(format!(
                "datetime_parse: tz_offset_hours {tz_offset_hours} out of range"
            ))
        })?;
        // Determine reference time.
        let reference: DateTime<FixedOffset> = if let Some(r_str) =
            args.get("reference").and_then(|v| v.as_str())
        {
            DateTime::parse_from_rfc3339(r_str)
                .map_err(|e| Error::invalid(format!("datetime_parse: bad reference: {e}")))?
                .with_timezone(&offset)
        } else {
            Utc::now().with_timezone(&offset)
        };

        let parsed = parse_natural(input, reference, offset).ok_or_else(|| {
            Error::invalid(format!(
                "datetime_parse: could not parse {input:?}. Supported patterns: \
                'today/yesterday/tomorrow/now', 'N days/weeks/hours/months ago', \
                'in N days/...', 'last/next <weekday>', ISO 8601, YYYY-MM-DD."
            ))
        })?;

        let weekday = match parsed.weekday() {
            Weekday::Mon => "Mon",
            Weekday::Tue => "Tue",
            Weekday::Wed => "Wed",
            Weekday::Thu => "Thu",
            Weekday::Fri => "Fri",
            Weekday::Sat => "Sat",
            Weekday::Sun => "Sun",
        };

        Ok(json!({
            "iso8601": parsed.to_rfc3339(),
            "date": format!("{:04}-{:02}-{:02}", parsed.year(), parsed.month(), parsed.day()),
            "weekday": weekday,
            "unix": parsed.timestamp(),
        }))
    }
}

fn parse_natural(
    raw: &str,
    reference: DateTime<FixedOffset>,
    offset: FixedOffset,
) -> Option<DateTime<FixedOffset>> {
    let s = raw.trim().to_lowercase();
    if s.is_empty() {
        return None;
    }

    match s.as_str() {
        "now" => return Some(reference),
        "today" => return Some(start_of_day(reference)),
        "yesterday" => return Some(start_of_day(reference - Duration::days(1))),
        "tomorrow" => return Some(start_of_day(reference + Duration::days(1))),
        _ => {}
    }

    // Try RFC 3339 (full ISO 8601 with timezone).
    if let Ok(dt) = DateTime::parse_from_rfc3339(raw.trim()) {
        return Some(dt);
    }
    // Try ISO 8601 without timezone — interpret in offset.
    if let Ok(dt) = NaiveDateTime::parse_from_str(raw.trim(), "%Y-%m-%dT%H:%M:%S") {
        return offset.from_local_datetime(&dt).single();
    }
    // Try YYYY-MM-DD date-only — start of day in offset.
    if let Ok(date) = NaiveDate::parse_from_str(raw.trim(), "%Y-%m-%d") {
        let dt = NaiveDateTime::new(
            date,
            NaiveTime::from_hms_opt(0, 0, 0).expect("0:0:0 is always valid"),
        );
        return offset.from_local_datetime(&dt).single();
    }

    // "N {unit} ago" / "in N {unit}"
    if let Some(d) = parse_relative_offset(&s) {
        return Some(reference + d);
    }

    // "last <weekday>" / "next <weekday>" / bare "<weekday>"
    if let Some(dt) = parse_weekday_phrase(&s, reference) {
        return Some(dt);
    }

    None
}

fn start_of_day(dt: DateTime<FixedOffset>) -> DateTime<FixedOffset> {
    dt.with_hour(0)
        .and_then(|d| d.with_minute(0))
        .and_then(|d| d.with_second(0))
        .and_then(|d| d.with_nanosecond(0))
        .unwrap_or(dt)
}

/// Parse `"N <unit> ago"` or `"in N <unit>"` into a signed Duration.
fn parse_relative_offset(s: &str) -> Option<Duration> {
    // "N <unit> ago"
    if let Some(rest) = s.strip_suffix(" ago") {
        let (n, unit) = split_n_unit(rest)?;
        let d = unit_duration(n, unit)?;
        return Some(-d);
    }
    // "in N <unit>"
    if let Some(rest) = s.strip_prefix("in ") {
        let (n, unit) = split_n_unit(rest)?;
        return unit_duration(n, unit);
    }
    None
}

/// Parse `"<integer> <unit-word>"` (handles plural "days/weeks/...").
fn split_n_unit(s: &str) -> Option<(i64, &str)> {
    let mut parts = s.splitn(2, ' ');
    let n: i64 = parts.next()?.parse().ok()?;
    let unit = parts.next()?.trim();
    Some((n, unit))
}

fn unit_duration(n: i64, unit: &str) -> Option<Duration> {
    // Strip trailing 's' for plurals: "days" -> "day".
    let u = unit.strip_suffix('s').unwrap_or(unit);
    match u {
        "second" | "sec" => Some(Duration::seconds(n)),
        "minute" | "min" => Some(Duration::minutes(n)),
        "hour" | "hr" => Some(Duration::hours(n)),
        "day" => Some(Duration::days(n)),
        "week" | "wk" => Some(Duration::weeks(n)),
        // Months and years are approximate (calendar arithmetic
        // would need chrono::Months, but the caller's contract is
        // ~30/365 days per unit — explicit in the docstring).
        "month" | "mo" => Some(Duration::days(n * 30)),
        "year" | "yr" => Some(Duration::days(n * 365)),
        _ => None,
    }
}

/// Parse `"last <weekday>"` / `"next <weekday>"` / bare `"<weekday>"`.
///
/// Bare weekday is interpreted as "the most recent past <weekday>",
/// matching English-speaking convention ("on Monday" usually means
/// the recent one when referring to the past). For future-direction
/// preference, callers should use `"next monday"` explicitly.
fn parse_weekday_phrase(
    s: &str,
    reference: DateTime<FixedOffset>,
) -> Option<DateTime<FixedOffset>> {
    let (direction, day_name): (i32, &str) = if let Some(rest) = s.strip_prefix("last ") {
        (-1, rest)
    } else if let Some(rest) = s.strip_prefix("next ") {
        (1, rest)
    } else {
        // Bare weekday → past (most recent <weekday>).
        (-1, s)
    };
    let target = parse_weekday_name(day_name)?;
    let ref_wd = reference.weekday();
    // days_offset = signed delta to land on target weekday.
    let target_idx = target.num_days_from_monday() as i32;
    let ref_idx = ref_wd.num_days_from_monday() as i32;
    let raw_delta = target_idx - ref_idx;
    let delta = if direction > 0 {
        // "next" — strictly forward; if same day, jump 7.
        if raw_delta <= 0 {
            raw_delta + 7
        } else {
            raw_delta
        }
    } else {
        // "last" / bare — strictly backward; if same day, jump -7.
        if raw_delta >= 0 {
            raw_delta - 7
        } else {
            raw_delta
        }
    };
    Some(start_of_day(reference + Duration::days(delta as i64)))
}

fn parse_weekday_name(s: &str) -> Option<Weekday> {
    match s.trim() {
        "monday" | "mon" => Some(Weekday::Mon),
        "tuesday" | "tue" | "tues" => Some(Weekday::Tue),
        "wednesday" | "wed" => Some(Weekday::Wed),
        "thursday" | "thu" | "thur" | "thurs" => Some(Weekday::Thu),
        "friday" | "fri" => Some(Weekday::Fri),
        "saturday" | "sat" => Some(Weekday::Sat),
        "sunday" | "sun" => Some(Weekday::Sun),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: Wed 2026-04-29 12:00 UTC. Picked because it's
    /// mid-week so weekday-relative tests can probe both directions
    /// without hitting weekend wraparound.
    fn ref_str() -> &'static str {
        "2026-04-29T12:00:00+00:00"
    }

    async fn run(input: &str) -> Value {
        let t = DatetimeParseTool::new();
        t.run(json!({"input": input, "reference": ref_str()}))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn today_returns_start_of_reference_day() {
        let v = run("today").await;
        assert_eq!(v.get("date").unwrap(), "2026-04-29");
        assert!(v
            .get("iso8601")
            .unwrap()
            .as_str()
            .unwrap()
            .starts_with("2026-04-29T00:00:00"));
    }

    #[tokio::test]
    async fn yesterday() {
        let v = run("yesterday").await;
        assert_eq!(v.get("date").unwrap(), "2026-04-28");
        assert_eq!(v.get("weekday").unwrap(), "Tue");
    }

    #[tokio::test]
    async fn tomorrow() {
        let v = run("tomorrow").await;
        assert_eq!(v.get("date").unwrap(), "2026-04-30");
        assert_eq!(v.get("weekday").unwrap(), "Thu");
    }

    #[tokio::test]
    async fn now_returns_full_reference_time() {
        let v = run("now").await;
        // "now" preserves time-of-day.
        assert!(v
            .get("iso8601")
            .unwrap()
            .as_str()
            .unwrap()
            .starts_with("2026-04-29T12:00:00"));
    }

    #[tokio::test]
    async fn n_days_ago() {
        let v = run("3 days ago").await;
        assert_eq!(v.get("date").unwrap(), "2026-04-26");
    }

    #[tokio::test]
    async fn n_weeks_ago() {
        let v = run("2 weeks ago").await;
        assert_eq!(v.get("date").unwrap(), "2026-04-15");
    }

    #[tokio::test]
    async fn n_hours_ago() {
        let v = run("5 hours ago").await;
        // 12:00 - 5h = 07:00 same day.
        assert!(v
            .get("iso8601")
            .unwrap()
            .as_str()
            .unwrap()
            .starts_with("2026-04-29T07:00:00"));
    }

    #[tokio::test]
    async fn in_n_days() {
        let v = run("in 7 days").await;
        assert_eq!(v.get("date").unwrap(), "2026-05-06");
    }

    #[tokio::test]
    async fn last_monday() {
        // Wed 2026-04-29 → "last monday" = 2026-04-27.
        let v = run("last monday").await;
        assert_eq!(v.get("date").unwrap(), "2026-04-27");
        assert_eq!(v.get("weekday").unwrap(), "Mon");
    }

    #[tokio::test]
    async fn next_friday() {
        // Wed 2026-04-29 → "next friday" = 2026-05-01.
        let v = run("next friday").await;
        assert_eq!(v.get("date").unwrap(), "2026-05-01");
        assert_eq!(v.get("weekday").unwrap(), "Fri");
    }

    #[tokio::test]
    async fn last_wednesday_jumps_back_a_week() {
        // Reference IS a Wednesday — "last wednesday" should go back
        // 7 days, not return today.
        let v = run("last wednesday").await;
        assert_eq!(v.get("date").unwrap(), "2026-04-22");
    }

    #[tokio::test]
    async fn next_wednesday_jumps_forward_a_week() {
        let v = run("next wednesday").await;
        assert_eq!(v.get("date").unwrap(), "2026-05-06");
    }

    #[tokio::test]
    async fn bare_weekday_is_most_recent_past() {
        // "tuesday" alone → most recent past Tuesday (the day before
        // a Wed reference).
        let v = run("tuesday").await;
        assert_eq!(v.get("date").unwrap(), "2026-04-28");
    }

    #[tokio::test]
    async fn iso8601_passes_through() {
        let v = run("2025-12-25T14:30:00+05:00").await;
        // Output preserves the input's components.
        let iso = v.get("iso8601").unwrap().as_str().unwrap();
        assert!(iso.starts_with("2025-12-25T14:30:00"));
    }

    #[tokio::test]
    async fn yyyy_mm_dd_passes_through() {
        let v = run("2024-07-04").await;
        assert_eq!(v.get("date").unwrap(), "2024-07-04");
        assert_eq!(v.get("weekday").unwrap(), "Thu"); // Indep day 2024 was a Thursday
    }

    #[tokio::test]
    async fn unsupported_string_errors() {
        let t = DatetimeParseTool::new();
        let r = t.run(json!({"input": "blargh-not-a-date", "reference": ref_str()})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn missing_input_errors() {
        let t = DatetimeParseTool::new();
        let r = t.run(json!({})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn tz_offset_applied_to_yyyy_mm_dd() {
        // YYYY-MM-DD parsed with tz_offset_hours = 5.5 (India).
        let t = DatetimeParseTool::new();
        let v = t
            .run(json!({
                "input": "2025-01-01",
                "reference": ref_str(),
                "tz_offset_hours": 5.5,
            }))
            .await
            .unwrap();
        let iso = v.get("iso8601").unwrap().as_str().unwrap();
        // 00:00 in IST → matches "+05:30" zone, NOT UTC.
        assert!(iso.starts_with("2025-01-01T00:00:00+05:30"), "iso={iso}");
    }

    #[tokio::test]
    async fn defaults_to_now_when_no_reference() {
        let t = DatetimeParseTool::new();
        let v = t.run(json!({"input": "today"})).await.unwrap();
        // We can't assert the exact date since "now" varies, but
        // the output should be valid ISO 8601 and "today" should
        // be one of the date strings produced by Utc::now().
        let date = v.get("date").unwrap().as_str().unwrap();
        assert!(date.starts_with("20"));
        assert_eq!(date.len(), 10);
    }
}
