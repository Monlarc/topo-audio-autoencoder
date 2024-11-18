# Topological Audio Encoder: WORK IN PROGRESS

A neural encoder that learns simplicial complex representations of audio signals using Hard Concrete distributions and topological constraints.

## Overview

This project implements a neural network encoder that converts audio signals into simplicial complex representations. The key components are:

1. **Audio Processing**
   - Processes multiple frequency bands independently
   - Cross-band integration for global features
   - Progressive temporal reduction

2. **Sparsification**
   - Uses Hard Concrete distribution for learnable sparsification
   - Continuous relaxation of binary random variables
   - Straight-Through Estimator (STE) for gradient propagation
   - Learned parameters: temperature, stretching (gamma/zeta), and location bias

3. **Topological Constraints**
   - Enforces valid simplicial complex structure
   - Hierarchical constraints between dimensions:
     - Edges require vertices
     - Triangles require edges
     - Tetrahedra require triangles
   - Implemented through geometric mean constraints

## Architecture Details

### Audio Encoder
