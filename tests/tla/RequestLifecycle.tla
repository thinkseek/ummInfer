---- MODULE RequestLifecycle ----
EXTENDS Naturals

CONSTANTS MaxOutputs, MaxOptionalLegs

VARIABLES state, phase, outputs, started, abortAccepted,
          vlmDone, legsRemaining, legErrors

vars == <<state, phase, outputs, started, abortAccepted,
          vlmDone, legsRemaining, legErrors>>

TerminalStates == {"completed", "failed", "aborted"}
ActivePhases == {"asr", "vlm", "outputs"}

TypeOK ==
    /\ state \in {"idle", "submitted", "running"} \cup TerminalStates
    /\ phase \in {"idle"} \cup ActivePhases \cup {"done"}
    /\ outputs \in 0..MaxOutputs
    /\ started \in BOOLEAN
    /\ abortAccepted \in BOOLEAN
    /\ vlmDone \in BOOLEAN
    /\ legsRemaining \in 0..MaxOptionalLegs
    /\ legErrors \in 0..MaxOptionalLegs

Init ==
    /\ state = "idle"
    /\ phase = "idle"
    /\ outputs = 0
    /\ started = FALSE
    /\ abortAccepted = FALSE
    /\ vlmDone = FALSE
    /\ legsRemaining = 0
    /\ legErrors = 0

Submit(hasAudio) ==
    /\ state = "idle"
    /\ state' = "submitted"
    /\ phase' = IF hasAudio THEN "asr" ELSE "vlm"
    /\ UNCHANGED <<outputs, started, abortAccepted, vlmDone, legsRemaining, legErrors>>

Start ==
    /\ state = "submitted"
    /\ state' = "running"
    /\ started' = TRUE
    /\ UNCHANGED <<phase, outputs, abortAccepted, vlmDone, legsRemaining, legErrors>>

AsrCompleteToVlm ==
    /\ state = "running"
    /\ phase = "asr"
    /\ ~abortAccepted
    /\ phase' = "vlm"
    /\ UNCHANGED <<state, outputs, started, abortAccepted, vlmDone, legsRemaining, legErrors>>

Emit ==
    /\ state = "running"
    /\ phase \in {"asr", "vlm", "outputs"}
    /\ outputs < MaxOutputs
    /\ outputs' = outputs + 1
    /\ UNCHANGED <<state, phase, started, abortAccepted, vlmDone, legsRemaining, legErrors>>

\* VLM completes with 0 optional legs -> immediate terminal completion.
VlmCompleteNoOptionalLegs ==
    /\ state = "running"
    /\ phase = "vlm"
    /\ ~abortAccepted
    /\ state' = "completed"
    /\ phase' = "done"
    /\ vlmDone' = TRUE
    /\ UNCHANGED <<outputs, started, abortAccepted, legsRemaining, legErrors>>

\* VLM barrier fans out k optional output legs (image / tts).
VlmCompleteStartOutputs(k) ==
    /\ state = "running"
    /\ phase = "vlm"
    /\ ~abortAccepted
    /\ k \in 1..MaxOptionalLegs
    /\ phase' = "outputs"
    /\ vlmDone' = TRUE
    /\ legsRemaining' = k
    /\ UNCHANGED <<state, outputs, started, abortAccepted, legErrors>>

\* An optional leg completes or fails (emitting leg_error, never failing the
\* primary VLM answer). When the last leg drains, the request settles.
SettleOptionalLeg(ok) ==
    /\ state = "running"
    /\ phase = "outputs"
    /\ legsRemaining > 0
    /\ legsRemaining' = legsRemaining - 1
    /\ legErrors' = IF ok THEN legErrors ELSE legErrors + 1
    /\ IF legsRemaining = 1
       THEN /\ phase' = "done"
            /\ state' = IF abortAccepted THEN "aborted" ELSE "completed"
       ELSE /\ UNCHANGED <<state, phase>>
    /\ UNCHANGED <<outputs, started, abortAccepted, vlmDone>>

\* Primary ASR or VLM failure fails the entire unified request.
PrimaryFail ==
    /\ state = "running"
    /\ phase \in {"asr", "vlm"}
    /\ state' = "failed"
    /\ phase' = "done"
    /\ legsRemaining' = 0
    /\ UNCHANGED <<outputs, started, abortAccepted, vlmDone, legErrors>>

\* Client/timeout cancellation settles the request as aborted even if optional
\* legs were in flight.
Abort ==
    /\ state \in {"submitted", "running"}
    /\ state' = "aborted"
    /\ phase' = "done"
    /\ abortAccepted' = TRUE
    /\ legsRemaining' = 0
    /\ UNCHANGED <<outputs, started, vlmDone, legErrors>>

TerminalStutter == state \in TerminalStates /\ UNCHANGED vars

Next ==
    \/ \E hasAudio \in BOOLEAN : Submit(hasAudio)
    \/ Start
    \/ AsrCompleteToVlm
    \/ Emit
    \/ VlmCompleteNoOptionalLegs
    \/ \E k \in 1..MaxOptionalLegs : VlmCompleteStartOutputs(k)
    \/ \E ok \in BOOLEAN : SettleOptionalLeg(ok)
    \/ PrimaryFail
    \/ Abort
    \/ TerminalStutter

Spec == Init /\ [][Next]_vars

NoOutputBeforeStart == outputs > 0 => started
BoundedOutput == outputs <= MaxOutputs
NoSuccessAfterAcceptedAbort == state = "completed" => ~abortAccepted
OutputsPhaseRequiresVlmDone == phase = "outputs" => vlmDone
CompletedRequiresDrainedLegs == state = "completed" => legsRemaining = 0
LegErrorsPreserveCompletion ==
    (legErrors > 0 /\ state \in TerminalStates /\ ~abortAccepted) => state = "completed"
TerminalStep == state \in TerminalStates => UNCHANGED vars
TerminalStatesCannotEmitOrTransition == [][TerminalStep]_vars

====
