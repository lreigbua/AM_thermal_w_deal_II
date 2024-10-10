# Additive Manufacturing thermal modeling with deal.II

Using the deal.II library, the provided code solves the transient heat transfer equation with adaptive mesh refinement based on the Kelly error estimator.
Created by modifying Step-26 of Deal.II tutorial (https://www.dealii.org/current/doxygen/deal.II/step_26.html).

Includes the following modifications:

- Contains inactive elements that get activated to simulate the printing of new layers, also adapting the cooling BCs to the new mesh.
- Modifies the thermal load BCs to move the heat source in time simulating a moving laser.

![Gif_output_vtk](https://github.com/lreigbua/AM_thermal_w_deal_II/assets/93150422/d92be467-f46d-4933-9494-1c94287f9886)
