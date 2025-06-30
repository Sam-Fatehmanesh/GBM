# Generative Brain Model: Neural Activity Simulation and Prediction

This is a research project focused on modeling and predicting neural spike activity patterns in brain imaging data. 

## Project Overview

This project aims to:
1. Process neural spike data from multiple subjects
2. Create autoencoder models to compress and reconstruct neural activity patterns
3. Implement a Generative Brain Model (GBM) that can predict future neural activity states
4. Provide visualization tools for analyzing model performance


MIT License

Copyright (c) 2023 BrainSim Project Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


python3 -m uvicorn GenerativeBrainModel.webapp.backend:app --reload