//! Independently authored bounded event broker, containing no sensor truth.
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    pub id: u64,
    pub capture_ns: u64,
    pub available_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueueError {
    ZeroCapacity,
    InvalidCapacity,
    InvalidIdBudget,
    IdBudgetExhausted,
    ArrivalBeforeCapture,
    DuplicateId,
    Full,
    PastArrival,
    TimeWentBackwards,
}

pub struct EventQueue {
    capacity: usize,
    max_ids: usize,
    now_ns: u64,
    pending: BTreeMap<(u64, u64), Event>,
    seen: HashSet<u64>,
}

impl EventQueue {
    pub fn new(capacity: usize) -> Result<Self, QueueError> {
        Self::with_id_budget(capacity, 1_000_000)
    }

    pub fn with_id_budget(capacity: usize, max_ids: usize) -> Result<Self, QueueError> {
        if capacity == 0 {
            return Err(QueueError::ZeroCapacity);
        }
        if capacity > 1_000_000 {
            return Err(QueueError::InvalidCapacity);
        }
        if max_ids < capacity || max_ids > 10_000_000 {
            return Err(QueueError::InvalidIdBudget);
        }
        Ok(Self {
            capacity,
            max_ids,
            now_ns: 0,
            pending: BTreeMap::new(),
            seen: HashSet::new(),
        })
    }

    pub fn push(&mut self, event: Event) -> Result<(), QueueError> {
        self.push_batch(&[event])
    }

    /// The whole batch is validated before any state or identity is consumed.
    pub fn push_batch(&mut self, events: &[Event]) -> Result<(), QueueError> {
        let mut ids = HashSet::new();
        for event in events {
            if event.available_ns < event.capture_ns {
                return Err(QueueError::ArrivalBeforeCapture);
            }
            if self.seen.contains(&event.id) || !ids.insert(event.id) {
                return Err(QueueError::DuplicateId);
            }
            if event.available_ns < self.now_ns {
                return Err(QueueError::PastArrival);
            }
        }
        if events.len() > self.max_ids - self.seen.len() {
            return Err(QueueError::IdBudgetExhausted);
        }
        if events.len() > self.capacity - self.pending.len() {
            return Err(QueueError::Full);
        }
        for event in events {
            self.seen.insert(event.id);
            self.pending
                .insert((event.available_ns, event.id), event.clone());
        }
        Ok(())
    }

    pub fn release(&mut self, now_ns: u64) -> Result<Vec<Event>, QueueError> {
        if now_ns < self.now_ns {
            return Err(QueueError::TimeWentBackwards);
        }
        self.now_ns = now_ns;
        let mut ready = Vec::new();
        while self
            .pending
            .first_key_value()
            .is_some_and(|(k, _)| k.0 <= now_ns)
        {
            if let Some((_, event)) = self.pending.pop_first() {
                ready.push(event);
            }
        }
        Ok(ready)
    }

    pub fn next_available_ns(&self) -> Option<u64> {
        self.pending.first_key_value().map(|(k, _)| k.0)
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(id: u64, cap: u64, avail: u64) -> Event {
        Event {
            id,
            capture_ns: cap,
            available_ns: avail,
        }
    }
    #[test]
    fn no_early_release() {
        let mut q = EventQueue::new(3).unwrap();
        q.push(e(1, 1, 10)).unwrap();
        assert!(q.release(9).unwrap().is_empty());
        assert_eq!(q.release(10).unwrap()[0].id, 1);
    }
    #[test]
    fn deterministic_arrival_order() {
        let mut q = EventQueue::new(3).unwrap();
        q.push_batch(&[e(7, 0, 20), e(4, 5, 10), e(3, 1, 10)])
            .unwrap();
        let ids: Vec<_> = q.release(20).unwrap().iter().map(|e| e.id).collect();
        assert_eq!(ids, vec![3, 4, 7]);
    }
    #[test]
    fn reject_invalid_capture() {
        let mut q = EventQueue::new(2).unwrap();
        assert_eq!(q.push(e(1, 10, 9)), Err(QueueError::ArrivalBeforeCapture));
    }
    #[test]
    fn duplicate_even_after_release() {
        let mut q = EventQueue::new(2).unwrap();
        q.push(e(1, 0, 1)).unwrap();
        q.release(1).unwrap();
        assert_eq!(q.push(e(1, 1, 2)), Err(QueueError::DuplicateId));
    }
    #[test]
    fn bounded_capacity_does_not_consume_id() {
        let mut q = EventQueue::new(1).unwrap();
        q.push(e(1, 0, 2)).unwrap();
        assert_eq!(q.push(e(2, 0, 2)), Err(QueueError::Full));
        q.release(2).unwrap();
        assert!(q.push(e(2, 2, 3)).is_ok());
    }
    #[test]
    fn monotonic_time() {
        let mut q = EventQueue::new(1).unwrap();
        q.release(10).unwrap();
        assert_eq!(q.release(9), Err(QueueError::TimeWentBackwards));
        assert_eq!(q.push(e(1, 0, 9)), Err(QueueError::PastArrival));
    }
    #[test]
    fn bounded_lifetime_ids() {
        let mut q = EventQueue::with_id_budget(1, 1).unwrap();
        q.push(e(1, 0, 1)).unwrap();
        q.release(1).unwrap();
        assert_eq!(q.push(e(2, 1, 2)), Err(QueueError::IdBudgetExhausted));
    }
    #[test]
    fn next_available_time() {
        let mut q = EventQueue::new(1).unwrap();
        assert_eq!(q.next_available_ns(), None);
        q.push(e(1, 0, 5)).unwrap();
        assert_eq!(q.next_available_ns(), Some(5));
    }
    #[test]
    fn capacity_limits() {
        assert!(matches!(EventQueue::new(0), Err(QueueError::ZeroCapacity)));
        assert!(matches!(
            EventQueue::new(1_000_001),
            Err(QueueError::InvalidCapacity)
        ));
        assert!(matches!(
            EventQueue::with_id_budget(2, 1),
            Err(QueueError::InvalidIdBudget)
        ));
    }
    #[test]
    fn batch_reject_is_atomic() {
        let mut q = EventQueue::new(3).unwrap();
        assert_eq!(
            q.push_batch(&[e(1, 0, 5), e(2, 8, 7)]),
            Err(QueueError::ArrivalBeforeCapture)
        );
        assert!(q.is_empty());
        q.push(e(1, 0, 5)).unwrap();
    }
    #[test]
    fn batch_internal_duplicate_is_atomic() {
        let mut q = EventQueue::new(3).unwrap();
        assert_eq!(
            q.push_batch(&[e(1, 0, 5), e(1, 0, 6)]),
            Err(QueueError::DuplicateId)
        );
        assert!(q.is_empty());
    }
    #[test]
    fn batch_overflow_is_atomic() {
        let mut q = EventQueue::new(1).unwrap();
        assert_eq!(
            q.push_batch(&[e(1, 0, 5), e(2, 0, 6)]),
            Err(QueueError::Full)
        );
        q.push(e(1, 0, 5)).unwrap();
    }
    #[test]
    fn maximum_integer_does_not_overflow() {
        let mut q = EventQueue::new(2).unwrap();
        q.push(e(u64::MAX, u64::MAX, u64::MAX)).unwrap();
        assert_eq!(q.release(u64::MAX).unwrap()[0].id, u64::MAX);
    }
}
