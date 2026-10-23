//! Line protocol for parity checks and optional language-neutral integration.
//! P <id> <capture_ns> <available_ns> => OK; R <now_ns> => READY <comma IDs>.
use isac_event_core::{Event, EventQueue};
use std::io::{self, BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut queue=EventQueue::new(4096).expect("positive capacity");
    let stdin=io::stdin(); let mut stdout=io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line=line?;
        let p:Vec<&str>=line.split_whitespace().collect();
        let response:Result<String,String>=(|| {
            let parse=|i:usize| p.get(i).ok_or("missing field")?.parse::<u64>().map_err(|_|"bad integer");
            match p.first().copied() {
                Some("P") if p.len()==4 => {
                    let e=Event{id:parse(1)?,capture_ns:parse(2)?,available_ns:parse(3)?};
                    queue.push(e).map_err(|e|format!("{e:?}"))?;
                    Ok("OK".to_string())
                }
                Some("R") if p.len()==2 => {
                    let result=queue.release(parse(1)?).map_err(|e|format!("{e:?}"))?;
                    Ok(format!("READY {}",result.iter().map(|e|e.id.to_string()).collect::<Vec<_>>().join(",")))
                }
                _=>Err("expected P id capture_ns available_ns or R now_ns".to_string()),
            }
        })();
        match response { Ok(s)=>writeln!(stdout,"{s}")?,Err(e)=>writeln!(stdout,"ERR {e}")? }
        stdout.flush()?;
    }
    Ok(())
}
