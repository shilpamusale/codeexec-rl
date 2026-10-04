# Deferred work

Items live here so they stay out of the build. Nothing on this list is started until the core result (Phases 0–5) is complete.

Priority order:

1. **V2 dose-response.** V1 with a reduced visible-test subset. Fewer visible tests should induce a larger gap if H1 holds, adding a dose-response dimension to the result.
2. **PPO comparison arm.** One arm under PPO against the GRPO default (ADR-003), so the algorithm choice is evidenced rather than asserted.
3. **H2 density suite.** Sparse vs shaped, with potential-based shaping as a possible safe-shaping arm. See THESIS.md.
4. **Hypothesis property-test canaries.** Supplementary held-out probes alongside EvalPlus (ADR-004).
5. **gVisor hardening.** Evaluated in Phase 1; implementing it is separate work.
6. **Multi-step episodes.** Iterative editing requires an edit representation, state serialization, and a per-step credit story.
7. **Repo-level tasks.** Multi-file engineering. A different project, honestly.
8. **Flaky-test detection.** MBPP function-level tests are deterministic, so this solves a problem this project does not have. Revisit only if a non-deterministic task source is adopted.
9. **Pool cancellation.** Clean cancellation of in-flight executions (and teardown of their containers) on interruption. The `finally` block already prevents leaked containers on normal and timeout paths; cancellation adds teardown on interrupt, a narrow case. Fiddly with thread-wrapped blocking calls — deferred until a real need (e.g. interactive early-stop of training) appears.
10. **Pool backpressure.** A bounded work queue, for unbounded/streaming submission. Current usage hands the pool a known, finite batch, so the concurrency semaphore already suffices; backpressure solves a load pattern this project does not produce.