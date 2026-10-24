//! Bounded line protocol. P <id> <capture_ns> <available_ns>; R <now_ns>.
//! Malformed individual commands return ERR without mutation. Oversized or
//! invalid-UTF8 input terminates with a nonzero status rather than allocating.
use isac_event_core::{Event, EventQueue};
use std::io::{self, BufRead, Read, Write};

const MAX_LINE_BYTES: u64 = 256;

fn dispatch(queue: &mut EventQueue, line: &str) -> Result<String, String> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let parse = |i: usize| -> Result<u64, String> {
        let token = parts.get(i).ok_or("missing field")?;
        if token.is_empty() || !token.bytes().all(|c| c.is_ascii_digit()) {
            return Err("bad integer".to_owned());
        }
        token.parse::<u64>().map_err(|_| "bad integer".to_owned())
    };
    match parts.first().copied() {
        Some("P") if parts.len() == 4 => {
            let event = Event {
                id: parse(1)?,
                capture_ns: parse(2)?,
                available_ns: parse(3)?,
            };
            queue.push(event).map_err(|e| format!("{e:?}"))?;
            Ok("OK".to_owned())
        }
        Some("R") if parts.len() == 2 => {
            let ready = queue.release(parse(1)?).map_err(|e| format!("{e:?}"))?;
            let ids: Vec<String> = ready.iter().map(|e| e.id.to_string()).collect();
            Ok(format!("READY {}", ids.join(",")))
        }
        _ => Err("expected P id capture_ns available_ns or R now_ns".to_owned()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut queue = EventQueue::new(4096).expect("positive capacity");
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut raw = Vec::with_capacity(MAX_LINE_BYTES as usize + 1);
    loop {
        raw.clear();
        let size = (&mut reader)
            .take(MAX_LINE_BYTES + 1)
            .read_until(b'\n', &mut raw)?;
        if size == 0 {
            break;
        }
        if size > MAX_LINE_BYTES as usize {
            return Err("protocol line exceeds 256 bytes".into());
        }
        let line = std::str::from_utf8(&raw)?;
        match dispatch(&mut queue, line) {
            Ok(s) => writeln!(stdout, "{s}")?,
            Err(e) => writeln!(stdout, "ERR {e}")?,
        }
        stdout.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_rejects_non_decimal_and_extra_fields() {
        let mut q = EventQueue::new(5).unwrap();
        for command in ["P +1 0 1", "P -1 0 1", "P 1 0 1 extra", "P 1 NaN 2", "R"] {
            assert!(dispatch(&mut q, command).is_err());
            assert!(q.is_empty());
        }
    }

    #[test]
    fn protocol_duplicate_and_order() {
        let mut q = EventQueue::new(5).unwrap();
        assert_eq!(dispatch(&mut q, "P 2 0 10"), Ok("OK".to_owned()));
        assert_eq!(dispatch(&mut q, "P 1 0 10"), Ok("OK".to_owned()));
        assert_eq!(dispatch(&mut q, "R 9"), Ok("READY ".to_owned()));
        assert_eq!(dispatch(&mut q, "R 10"), Ok("READY 1,2".to_owned()));
        assert_eq!(dispatch(&mut q, "P 1 10 11"), Err("DuplicateId".to_owned()));
    }
}
