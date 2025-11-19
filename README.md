# Additive Manufacturing thermal modeling with deal.II

Using the deal.II library, the provided code solves the transient heat transfer equation with adaptive mesh refinement that keeps a refined mesh around a moving heat source.
Includes the following features:

- Contains inactive elements that get activated to simulate the printing of new layers, also adapting the cooling BCs to the new mesh.
- Modifies the thermal load BCs to move the heat source in time simulating a moving laser.
- Keeps a refined mesh around the heat source, where gradients are larger, improving significantly computational efficiency.

![animation-am_adaptive](https://github.com/user-attachments/assets/db861963-5fb2-4fcb-8dfa-1a7f9d75cac9)





