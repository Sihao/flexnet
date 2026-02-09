# Input-dependent flexible computation as an inductive bias for learning robust feature representations

**Sihao Lu¹, Dong Yin¹, Linnea Evanson¹, Maksim Lavrov¹, Iakov Kharitonov¹, Andriy S. Kozlov¹**

¹ Department of Bioengineering, Imperial College London

---

This repository contains the implementation for **Flexible Neurons**, exploring input-dependent flexible computation as an inductive bias to enhance the robustness of learned feature representations.

## Abstract
Object recognition requires balancing selectivity and invariance, a feat achieved in both biological and artificial systems through AND-like and OR-like operations. While biological neurons are capable of intrinsically switching between these operations, artificial units are typically hard-wired for a single computation. We address this discrepancy by introducing a deep learning layer where input strength flexibly governs the choice between selectivity and invariance. This flexible mechanism serves as an inductive bias predisposing the network towards low-frequency encoding, and a more smooth input loss landscape. We show that these properties confer robustness against algorithmic and adversarial corruptions. Furthermore, this approach enhances alignment with biological representations, offering a principled, biomimetic strategy for defensive optimization.

## Installation

```bash
pip install -r requirements.txt
```

## Acknowledgements

This work was conducted at the [Kozlov Lab](https://www.kozlovlab.com/).
Final code by Sihao Lu and Dong Yin, based on earlier implementations by Linnea Evanson, Maksim Lavrov, Iakov Kharitonov.
