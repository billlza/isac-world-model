# Provenance and conceptual sources

All project source files were independently authored for this research request.
No external repository files, pretrained weights, private experimental data, or
restricted model outputs were incorporated. Familiar CNN, GRU, Gaussian NLL and
Kalman filtering components are explicitly acknowledged as established methods.

Direct software dependencies: Python, NumPy, PyTorch, pytest (development), and
Rust standard library for the native core. Dependencies retain their own licenses.
No public license is chosen for this repository yet.

Conceptual background, not code copied into this implementation:

- Paul, Chiriyath and Bliss, *Survey of RF Communications and Sensing Convergence
  Research*, IEEE Access 2017, DOI 10.1109/ACCESS.2016.2639038. Joint design and
  dynamic time allocation motivate the resource accounting abstraction.
- Liu et al., *A Survey on Fundamental Limits of Integrated Sensing and
  Communication*, 2022, DOI 10.1109/COMST.2022.3149272. Separate sensing and
  communications metrics and their trade-offs.
- Yang, Wen and Jin, *Hybrid Active and Passive Sensing for SLAM in Wireless
  Communication Systems*, DOI 10.1109/JSAC.2022.3156630. Common physical features
  with uncertainty are a modelling reference; their SLAM algorithm is NOT implemented.
- Ha and Schmidhuber, *World Models*, https://worldmodels.github.io/.
- Zhou et al., *DINO-WM*, https://github.com/gaoyuezhou/dino_wm. Conceptual reference
  for latent prediction separated from image rendering; no DINO/other pretrained
  features or source code are used here.
- PyTorch documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
  and https://docs.pytorch.org/docs/stable/notes/mps.html.

Novelty requires a focused literature comparison; no first-ever claim is made.
