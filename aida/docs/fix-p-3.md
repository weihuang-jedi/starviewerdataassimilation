2. Introduce a Geopotential/Hydrostatic Consistency TermIf $p$ is drifting relative to $T$, the Ideal Gas Law constraint (loss_state_eq) might be too weak or fighting the MSE term. Raising weight_state_eq from $0.10 \rightarrow \mathbf{0.20}$ forces $p$, $T$, and $\rho$ to move in physical lockstep.


