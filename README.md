# train_with_reversibility

Reversible 20M-Parameter LLM on Google Colab (50M Tokens)

[![CI](https://github.com/thinkTinkerGreen/train_with_reversibility/actions/workflows/ci.yml/badge.svg)](https://github.com/thinkTinkerGreen/train_with_reversibility/actions/workflows/ci.yml)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

## Overview
This repository investigates pre-training a ~20.85M parameter decoder-only Transformer language model for 50M tokens on a Google Colab free-tier GPU (NVIDIA T4 16GB), comparing standard residual connections (forward-Euler) with time-reversible dynamics (midpoint, midpoint-$\theta$, leapfrog, Hamiltonian).

Detailed benchmarks and findings will be generated upon completion of Colab executions.
