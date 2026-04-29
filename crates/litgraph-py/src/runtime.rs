//! Single shared tokio runtime for Python-facing async work.
//!
//! We can't let Python create a new runtime for every call (spin-up is ~ms), and
//! holding the GIL while constructing one is worse. One multi-threaded runtime
//! lives for the module's lifetime; blocking methods `block_on` into it without
//! the GIL held.

use std::future::Future;

use once_cell::sync::OnceCell;
use tokio::runtime::{Handle, Runtime};

static RT: OnceCell<Runtime> = OnceCell::new();

pub fn rt() -> &'static Runtime {
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_cpus().max(2))
            .enable_all()
            .thread_name("litgraph-rt")
            .build()
            .expect("tokio runtime build failed")
    })
}

/// Block on a future from anywhere — including from inside an existing tokio
/// task (e.g. a Python `FunctionTool` invoked by the agent loop, which runs
/// under the agent's tokio task). If we're already on a tokio worker thread,
/// use `block_in_place` so the runtime can swap us out and continue driving
/// other tasks. Otherwise, fall back to the shared runtime's `block_on`.
///
/// Without this, calling `rt().block_on()` from a tool callback panics with
/// "Cannot start a runtime from within a runtime".
pub fn block_on_compat<F: Future>(fut: F) -> F::Output {
    match Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fut)),
        Err(_) => rt().block_on(fut),
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}
