# Growing Network Model

## Description

This project explores the Barabási-Albert (BA) network model, governed by three different attachment rules: Preferential Attachment (PA), Random Attachment (RA), and Existing Vertices (EV). The network is simulated in Python using Object-Oriented Programming (OOP) principles. Each model investigates the evolution of networks as they grow over time, with specific attention to their degree distributions and the characteristics of the network's largest degree.

The key features of this project include:

- **Preferential Attachment (PA):** Nodes are added to the network with a probability proportional to the node's degree.
- **Random Attachment (RA):** The probability of attaching a new node to an existing vertex is equal for all vertices.
- **Existing Vertices (EV):** A portion of new nodes is attached preferentially, the rest is attached randomly.

For each model, the theoretical degree probability distributions are derived and compared with the simulated results. The statistical tests show that the PA and RA models agree well with theory, with p-values of 1.000 in regions before the characteristic bump. However, the EV model does not align as closely with the theory, especially for larger network sizes where p-values fall below 0.05.

Additionally, a theoretical scaling of the largest degree is derived for each model, enabling data collapse by aligning the characteristic bumps in the degree distributions. The PA model yielded the best agreement with the theoretical predictions, while the EV model showed the poorest performance, suggesting that further refinement of its theoretical foundation is necessary.

For more information, read the report.

## Features

- Simulates three different network models: PA, RA, and EV
- Implements Object-Oriented Programming (OOP) for clean, reusable code
- Derives theoretical degree distributions for each model and compares them with simulation results
- Performs statistical tests to assess the quality of the theory against numerical results
- Analyzes the scaling of the largest degree and performs data collapse
