//! Minimal AWS event-stream frame parser. Just enough for Bedrock
//! `InvokeModelWithResponseStream`.
//!
//! # Wire format
//!
//! Each message:
//!
//! ```text
//! [4 bytes BE: total_length]
//! [4 bytes BE: headers_length]
//! [4 bytes BE: prelude_crc]   (CRC32 of first 8 bytes — not validated here)
//! [headers_length bytes: headers]
//! [total_length - headers_length - 16 bytes: payload]
//! [4 bytes BE: message_crc]   (CRC32 of all bytes except this CRC — not validated)
//! ```
//!
//! Headers (each):
//!
//! ```text
//! [1 byte: name_length]
//! [name_length bytes: name]
//! [1 byte: value_type]
//! [type-specific value bytes]
//! ```
//!
//! For Bedrock chunks we care about value_type 7 (string) which has a 2-byte
//! BE length prefix followed by UTF-8 bytes. Other value types are skipped
//! safely (we know their on-wire sizes).
//!
//! CRC validation is intentionally skipped — it's a robustness check, not
//! required for parsing. AWS clients re-validate on the wire layer; we trust
//! the TLS connection.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Frame {
    pub headers: HashMap<String, String>,
    pub payload: Vec<u8>,
}

#[derive(Debug)]
#[allow(dead_code)] // future error variants kept for API stability
pub enum ParseError {
    Incomplete,
    BadHeaderType(u8),
    BadUtf8,
    Truncated,
}

/// Parse one frame from the front of `buf`. Returns Ok(Some((frame, consumed)))
/// on success, Ok(None) if the buffer doesn't yet contain a complete frame,
/// or Err on a malformed frame.
pub fn parse_frame(buf: &[u8]) -> Result<Option<(Frame, usize)>, ParseError> {
    if buf.len() < 12 {
        return Ok(None);
    }
    let total_len = u32::from_be_bytes(buf[0..4].try_into().unwrap()) as usize;
    let headers_len = u32::from_be_bytes(buf[4..8].try_into().unwrap()) as usize;
    if buf.len() < total_len {
        return Ok(None);
    }
    if total_len < 16 + headers_len {
        return Err(ParseError::Truncated);
    }

    let headers_start = 12;
    let headers_end = headers_start + headers_len;
    let payload_end = total_len - 4;
    let headers = parse_headers(&buf[headers_start..headers_end])?;
    let payload = buf[headers_end..payload_end].to_vec();

    Ok(Some((Frame { headers, payload }, total_len)))
}

fn parse_headers(buf: &[u8]) -> Result<HashMap<String, String>, ParseError> {
    let mut out = HashMap::new();
    let mut i = 0;
    while i < buf.len() {
        if i + 1 > buf.len() { return Err(ParseError::Truncated); }
        let name_len = buf[i] as usize;
        i += 1;
        if i + name_len + 1 > buf.len() { return Err(ParseError::Truncated); }
        let name = std::str::from_utf8(&buf[i..i + name_len])
            .map_err(|_| ParseError::BadUtf8)?
            .to_string();
        i += name_len;
        let value_type = buf[i];
        i += 1;
        match value_type {
            7 => {
                // string: 2-byte BE length + UTF-8 bytes
                if i + 2 > buf.len() { return Err(ParseError::Truncated); }
                let val_len = u16::from_be_bytes([buf[i], buf[i + 1]]) as usize;
                i += 2;
                if i + val_len > buf.len() { return Err(ParseError::Truncated); }
                let val = std::str::from_utf8(&buf[i..i + val_len])
                    .map_err(|_| ParseError::BadUtf8)?
                    .to_string();
                i += val_len;
                out.insert(name, val);
            }
            // Skip-by-known-size for other types we don't need to interpret.
            // The full spec has 10 types — we cover only what Bedrock sends.
            0 | 1 => {} // boolean true/false — no value bytes
            2 => { i += 1; } // byte
            3 => { i += 2; } // short
            4 => { i += 4; } // integer
            5 => { i += 8; } // long
            6 => {
                // bytes: 2-byte len + bytes (skip)
                if i + 2 > buf.len() { return Err(ParseError::Truncated); }
                let n = u16::from_be_bytes([buf[i], buf[i + 1]]) as usize;
                i += 2 + n;
            }
            8 => { i += 8; } // timestamp (ms since epoch)
            9 => { i += 16; } // uuid
            other => return Err(ParseError::BadHeaderType(other)),
        }
        if i > buf.len() { return Err(ParseError::Truncated); }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal frame with one string header (`:event-type=chunk`)
    /// and a JSON payload, including dummy CRCs.
    fn build_chunk_frame(payload: &[u8]) -> Vec<u8> {
        let header_name = b":event-type";
        let header_value = b"chunk";
        let mut headers = Vec::new();
        headers.push(header_name.len() as u8);
        headers.extend_from_slice(header_name);
        headers.push(7u8); // string
        let vlen = (header_value.len() as u16).to_be_bytes();
        headers.extend_from_slice(&vlen);
        headers.extend_from_slice(header_value);

        let total_len = 12 + headers.len() + payload.len() + 4;
        let mut frame = Vec::with_capacity(total_len);
        frame.extend_from_slice(&(total_len as u32).to_be_bytes());
        frame.extend_from_slice(&(headers.len() as u32).to_be_bytes());
        frame.extend_from_slice(&[0u8; 4]); // dummy prelude CRC
        frame.extend_from_slice(&headers);
        frame.extend_from_slice(payload);
        frame.extend_from_slice(&[0u8; 4]); // dummy message CRC
        assert_eq!(frame.len(), total_len);
        frame
    }

    #[test]
    fn parses_single_frame() {
        let payload = br#"{"bytes":"abc"}"#;
        let frame = build_chunk_frame(payload);
        let (parsed, n) = parse_frame(&frame).unwrap().unwrap();
        assert_eq!(n, frame.len());
        assert_eq!(parsed.headers.get(":event-type"), Some(&"chunk".to_string()));
        assert_eq!(parsed.payload, payload);
    }

    #[test]
    fn returns_none_when_incomplete() {
        let payload = br#"{"bytes":"abc"}"#;
        let frame = build_chunk_frame(payload);
        assert!(parse_frame(&frame[..5]).unwrap().is_none());
        assert!(parse_frame(&frame[..frame.len() - 1]).unwrap().is_none());
    }

    #[test]
    fn parses_back_to_back_frames() {
        let f1 = build_chunk_frame(br#"{"bytes":"AAA"}"#);
        let f2 = build_chunk_frame(br#"{"bytes":"BBB"}"#);
        let mut buf = Vec::new();
        buf.extend_from_slice(&f1);
        buf.extend_from_slice(&f2);

        let (a, n1) = parse_frame(&buf).unwrap().unwrap();
        assert_eq!(a.payload, br#"{"bytes":"AAA"}"#);
        let (b, n2) = parse_frame(&buf[n1..]).unwrap().unwrap();
        assert_eq!(b.payload, br#"{"bytes":"BBB"}"#);
        assert_eq!(n1 + n2, buf.len());
    }
}
