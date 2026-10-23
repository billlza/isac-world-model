//! Independently authored event broker. No model, sensor truth or pretrained data.
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
        if capacity == 0 { return Err(QueueError::ZeroCapacity); }
        if max_ids < capacity { return Err(QueueError::InvalidIdBudget); }
        Ok(Self { capacity, max_ids, now_ns: 0, pending: BTreeMap::new(), seen: HashSet::new() })
    }
    pub fn push(&mut self, event: Event) -> Result<(), QueueError> {
        if event.available_ns < event.capture_ns { return Err(QueueError::ArrivalBeforeCapture); }
        if self.seen.contains(&event.id) { return Err(QueueError::DuplicateId); }
        if event.available_ns < self.now_ns { return Err(QueueError::PastArrival); }
        if self.seen.len() >= self.max_ids { return Err(QueueError::IdBudgetExhausted); }
        if self.pending.len() == self.capacity { return Err(QueueError::Full); }
        self.seen.insert(event.id);
        self.pending.insert((event.available_ns, event.id), event);
        Ok(())
    }
    pub fn release(&mut self, now_ns: u64) -> Result<Vec<Event>, QueueError> {
        if now_ns < self.now_ns { return Err(QueueError::TimeWentBackwards); }
        self.now_ns = now_ns;
        let mut ready = Vec::new();
        while self.pending.first_key_value().map(|(k, _)| k.0 <= now_ns).unwrap_or(false) {
            if let Some((_, event)) = self.pending.pop_first() { ready.push(event); }
        }
        Ok(ready)
    }
    pub fn next_available_ns(&self) -> Option<u64> { self.pending.first_key_value().map(|(k, _)| k.0) }
    pub fn len(&self) -> usize { self.pending.len() }
    pub fn is_empty(&self) -> bool { self.pending.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn e(id: u64, cap: u64, avail: u64) -> Event { Event { id, capture_ns: cap, available_ns: avail } }
    #[test] fn no_early_release() {
        let mut q = EventQueue::new(3).unwrap(); q.push(e(1, 1, 10)).unwrap();
        assert!(q.release(9).unwrap().is_empty()); assert_eq!(q.release(10).unwrap()[0].id, 1);
    }
    #[test] fn deterministic_arrival_order() {
        let mut q=EventQueue::new(3).unwrap(); q.push(e(7,0,20)).unwrap();
        q.push(e(4,5,10)).unwrap(); q.push(e(3,1,10)).unwrap();
        assert_eq!(q.release(20).unwrap().iter().map(|e|e.id).collect::<Vec<_>>(),vec![3,4,7]);
    }
    #[test] fn reject_invalid_capture() {
        let mut q=EventQueue::new(2).unwrap(); assert_eq!(q.push(e(1,10,9)),Err(QueueError::ArrivalBeforeCapture));
    }
    #[test] fn duplicate_even_after_release() {
        let mut q=EventQueue::new(2).unwrap(); q.push(e(1,0,1)).unwrap(); q.release(1).unwrap();
        assert_eq!(q.push(e(1,1,2)),Err(QueueError::DuplicateId));
    }
    #[test] fn bounded_capacity() {
        let mut q=EventQueue::new(1).unwrap(); q.push(e(1,0,2)).unwrap();
        assert_eq!(q.push(e(2,0,2)),Err(QueueError::Full)); q.release(2).unwrap();
        assert!(q.push(e(2,2,3)).is_ok());
    }
    #[test] fn monotonic_time() {
        let mut q=EventQueue::new(1).unwrap(); q.release(10).unwrap();
        assert_eq!(q.release(9),Err(QueueError::TimeWentBackwards));
        assert_eq!(q.push(e(1,0,9)),Err(QueueError::PastArrival));
    }
    #[test] fn bounded_lifetime_ids() {
        let mut q=EventQueue::with_id_budget(1,1).unwrap();q.push(e(1,0,1)).unwrap();q.release(1).unwrap();
        assert_eq!(q.push(e(2,1,2)),Err(QueueError::IdBudgetExhausted));
    }
    #[test] fn next_available_time() {
        let mut q=EventQueue::new(1).unwrap();assert_eq!(q.next_available_ns(),None);
        q.push(e(1,0,5)).unwrap();assert_eq!(q.next_available_ns(),Some(5));
    }
    #[test] fn zero_capacity() { assert!(matches!(EventQueue::new(0),Err(QueueError::ZeroCapacity))); }
}
