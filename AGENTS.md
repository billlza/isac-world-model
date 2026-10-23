# Research engineering constraints

Main model components must start from random initialization unless the user
explicitly changes the design. Baselines may use pretrained models, clearly labelled.
Do not copy restricted code, weights, model outputs, or private experimental data.

Keep capture time, availability time, estimated state time, and deadlines distinct.
No future truth or future measurements may enter online inputs or decisions.
Sensing actions affect observation/noise/cost, not exogenous target dynamics.
Report all baseline outcomes, failures, dropped packets, resource costs and timing.
Do not turn an empirical uncertainty score into a safety certificate.
Quantum is not implemented; require an explicit physical model before that claim.
Pin and record source/data/checkpoint/runtime versions. Preserve held-out episode
splits and separate training supervision from estimator inputs. Test before pushing.
Do not publish private research as public without an explicit user instruction.
