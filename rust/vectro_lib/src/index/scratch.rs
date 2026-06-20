//! Reusable per-thread scratch buffers for the HNSW beam search.
//!
//! `search_layer_impl` needs a "have I seen this node?" set.  The previous
//! implementation allocated a fresh `HashSet` on **every** layer call — i.e.
//! `1 + max_level` allocations per query plus a hash per probe.  Both HNSW
//! indexes now share a thread-local **epoch (generation) array** instead:
//!
//! * `marks[id] == generation` means "visited this epoch".  Marking is a single
//!   store and a check is a single load — no hashing, no probing.
//! * A new search/layer bumps `generation` ([`VisitedEpoch::begin`]) so the
//!   whole set is "cleared" in O(1) without touching memory.
//! * The backing `Vec<u32>` lives in a `thread_local!`, so it is allocated once
//!   per worker thread and reused across every subsequent query.  Concurrent
//!   `search(&self)` calls on different threads each get their own buffer, so
//!   the index stays `Sync` and lock-free.
//!
//! Trade-off vs. the old `HashSet`: steady-state scratch is `O(max N queried on
//! the thread)` rather than `O(nodes visited)`.  This is the standard ANN-engine
//! design (cf. FAISS `VisitedTable`) and is amortised to zero allocations per
//! query.

use std::cell::RefCell;

thread_local! {
    /// Per-thread visited set, reused across all queries on this thread.
    static VISITED: RefCell<VisitedEpoch> = const { RefCell::new(VisitedEpoch::new()) };
}

/// Generation-stamped visited set.  See the module docs.
pub(crate) struct VisitedEpoch {
    marks: Vec<u32>,
    generation: u32,
}

impl VisitedEpoch {
    const fn new() -> Self {
        Self { marks: Vec::new(), generation: 0 }
    }

    /// Begin a fresh epoch covering node ids `0..n`.
    ///
    /// Grows the backing store if needed and bumps the generation so every
    /// previously-marked id reads as unvisited.  On the (≈4-billion-query)
    /// generation wraparound, the marks are zeroed so stale `0` stamps cannot
    /// be mistaken for the current epoch.
    fn begin(&mut self, n: usize) {
        if self.marks.len() < n {
            self.marks.resize(n, 0);
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.iter_mut().for_each(|m| *m = 0);
            self.generation = 1;
        }
    }

    /// Mark `id` visited; returns `true` if it was newly visited this epoch
    /// (mirrors `HashSet::insert`).
    ///
    /// `id` must be `< n` from the most recent [`begin`](Self::begin) call.
    #[inline]
    pub(crate) fn visit(&mut self, id: usize) -> bool {
        if self.marks[id] == self.generation {
            false
        } else {
            self.marks[id] = self.generation;
            true
        }
    }
}

/// Run `f` with the thread-local visited set freshly reset for `n` nodes.
///
/// The borrow is held only for the duration of `f`; `f` must not recurse into
/// another `with_visited` call (HNSW beam search never does).
#[inline]
pub(crate) fn with_visited<R>(n: usize, f: impl FnOnce(&mut VisitedEpoch) -> R) -> R {
    VISITED.with(|cell| {
        let mut visited = cell.borrow_mut();
        visited.begin(n);
        f(&mut visited)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_epoch_sees_nothing_then_dedupes() {
        with_visited(8, |v| {
            assert!(v.visit(3), "first visit is new");
            assert!(!v.visit(3), "second visit is a duplicate");
            assert!(v.visit(5));
        });
        // A new epoch must forget the previous marks (same thread).
        with_visited(8, |v| {
            assert!(v.visit(3), "id 3 must be unvisited in the new epoch");
        });
    }

    #[test]
    fn grows_to_cover_larger_n() {
        with_visited(2, |v| assert!(v.visit(1)));
        with_visited(64, |v| {
            assert!(v.visit(63), "must cover grown range");
            assert!(!v.visit(63));
        });
    }

    #[test]
    fn generation_wraparound_resets_cleanly() {
        // Drive the generation to the brink of overflow, then cross it.
        VISITED.with(|cell| cell.borrow_mut().generation = u32::MAX - 1);
        with_visited(4, |v| assert!(v.visit(0))); // generation -> u32::MAX
        with_visited(4, |v| {
            // generation wrapped to 0 then reset to 1; marks were zeroed.
            assert!(v.visit(0), "post-wrap epoch must treat ids as unvisited");
            assert!(!v.visit(0));
        });
    }
}
