//! `CurrentTimeTool` — return the current date/time. Surprisingly
//! often needed by agents: any reasoning involving "today",
//! "next Tuesday", "two weeks from now", date-relative scheduling,
//! or comparing against a stored timestamp requires the agent to
//! know the current moment. LLMs don't know what time it is
//! unless something tells them.
//!
//! # Output
//!
//! - `iso8601` — RFC3339 timestamp.
//! - `unix` — seconds since epoch (i64).
//! - `weekday` — three-letter day name (`Mon`, `Tue`, …).
//! - `date` — ISO date (`YYYY-MM-DD`).
//! - `time` — wall-clock time (`HH:MM:SS`).
//! - `tz` — the timezone offset that was applied (`UTC` if none).
//!
//! # Args
//!
//! - `tz_offset_hours: Option<f64>` — apply a fixed UTC offset
//!   (`-7.0`, `5.5`, `0.0`). Optional; defaults to UTC.
//!
//! IANA timezone-name resolution (`America/Los_Angeles`) is NOT
//! supported here — that requires the heavy `chrono-tz` crate
//! and a baked-in DB. Callers needing tz-name support should
//! inject the offset explicitly or wrap this tool with their
//! own pre-resolution step.

use async_trait::async_trait;
use chrono::{Datelike, FixedOffset, TimeZone, Timelike, Utc};
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, Default)]
pub struct CurrentTimeTool;

impl CurrentTimeTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for CurrentTimeTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "current_time".into(),
            description: "Return the current date and time. Useful when reasoning about \
                'today', 'next Tuesday', or date-relative scheduling. Optionally \
                applies a fixed UTC offset (in hours, can be fractional like 5.5 for \
                IST). Returns ISO-8601 timestamp, Unix seconds, weekday, date, and time."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "tz_offset_hours": {
                        "type": "number",
                        "description": "UTC offset in hours (e.g. -7.0 for PDT, 5.5 for IST, 0 for UTC). Default: 0 (UTC)."
                    }
                }
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let now_utc = Utc::now();
        let unix = now_utc.timestamp();
        let offset_hours = args
            .get("tz_offset_hours")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        // Convert fractional hours to seconds, clamping to chrono's
        // FixedOffset bounds (±26 hours wide enough for every real tz).
        let offset_secs = (offset_hours * 3600.0).round() as i32;
        if offset_secs.abs() > 26 * 3600 {
            return Err(Error::invalid(
                "current_time: tz_offset_hours out of range (±26h)",
            ));
        }
        let offset = FixedOffset::east_opt(offset_secs)
            .ok_or_else(|| Error::invalid("current_time: invalid tz_offset_hours"))?;
        let local = offset.from_utc_datetime(&now_utc.naive_utc());
        let tz_label = if offset_secs == 0 {
            "UTC".to_string()
        } else {
            let sign = if offset_secs >= 0 { "+" } else { "-" };
            let abs = offset_secs.unsigned_abs();
            let h = abs / 3600;
            let m = (abs % 3600) / 60;
            format!("UTC{sign}{h:02}:{m:02}")
        };
        let weekday = match local.weekday() {
            chrono::Weekday::Mon => "Mon",
            chrono::Weekday::Tue => "Tue",
            chrono::Weekday::Wed => "Wed",
            chrono::Weekday::Thu => "Thu",
            chrono::Weekday::Fri => "Fri",
            chrono::Weekday::Sat => "Sat",
            chrono::Weekday::Sun => "Sun",
        };
        Ok(json!({
            "iso8601": local.to_rfc3339(),
            "unix": unix,
            "weekday": weekday,
            "date": format!("{:04}-{:02}-{:02}", local.year(), local.month(), local.day()),
            "time": format!("{:02}:{:02}:{:02}", local.hour(), local.minute(), local.second()),
            "tz": tz_label,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_full_metadata_in_utc_by_default() {
        let t = CurrentTimeTool::new();
        let v = t.run(json!({})).await.unwrap();
        assert!(v.get("iso8601").and_then(|x| x.as_str()).is_some());
        assert!(v.get("unix").and_then(|x| x.as_i64()).is_some());
        let wd = v.get("weekday").and_then(|x| x.as_str()).unwrap();
        assert!(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].contains(&wd));
        assert_eq!(v.get("tz").and_then(|x| x.as_str()), Some("UTC"));
        // date is YYYY-MM-DD, time is HH:MM:SS.
        let date = v.get("date").and_then(|x| x.as_str()).unwrap();
        assert_eq!(date.len(), 10);
        assert_eq!(&date[4..5], "-");
        let time = v.get("time").and_then(|x| x.as_str()).unwrap();
        assert_eq!(time.len(), 8);
    }

    #[tokio::test]
    async fn applies_positive_offset() {
        let t = CurrentTimeTool::new();
        // IST = UTC+5:30
        let v = t.run(json!({"tz_offset_hours": 5.5})).await.unwrap();
        assert_eq!(v.get("tz").and_then(|x| x.as_str()), Some("UTC+05:30"));
    }

    #[tokio::test]
    async fn applies_negative_offset() {
        let t = CurrentTimeTool::new();
        // PDT = UTC-7
        let v = t.run(json!({"tz_offset_hours": -7.0})).await.unwrap();
        assert_eq!(v.get("tz").and_then(|x| x.as_str()), Some("UTC-07:00"));
    }

    #[tokio::test]
    async fn out_of_range_offset_errors() {
        let t = CurrentTimeTool::new();
        let r = t.run(json!({"tz_offset_hours": 100.0})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn unix_and_iso_are_consistent() {
        let t = CurrentTimeTool::new();
        let v = t.run(json!({"tz_offset_hours": 0})).await.unwrap();
        let unix = v.get("unix").and_then(|x| x.as_i64()).unwrap();
        let iso = v.get("iso8601").and_then(|x| x.as_str()).unwrap();
        // Parse the ISO back and compare to the unix.
        let parsed = chrono::DateTime::parse_from_rfc3339(iso).unwrap();
        assert_eq!(parsed.timestamp(), unix);
    }

    #[tokio::test]
    async fn schema_has_required_fields() {
        let t = CurrentTimeTool::new();
        let s = t.schema();
        assert_eq!(s.name, "current_time");
        assert!(s.description.contains("date and time"));
        // tz_offset_hours is optional (not in `required`).
        let params = &s.parameters;
        assert_eq!(params.get("type").and_then(|v| v.as_str()), Some("object"));
    }
}
