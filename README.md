# On the Combination of Symbolic Dynamics and Dynamic Programming for Dynamic Systems

## Master’s Thesis - TU Munich - Chair of Data Processing

### Abstract

Investigation and understanding of nonlinear dynamic systems are crucial
in various applications and fields of research as describing chaotic
systems in physics , comprehend bio-molecular processes for designing
new medial drugs  or in engineering as for developing new
high-performance materials . Simultaneously, such systems’ global
dynamics are notoriously hard to represent, although differential
equations to describe such systems are well-known in some cases. A
standard tool for analyzing dynamic behavior over time are dynamic
simulations . However, these techniques often suffer from a very high
dimensional data space resulting in enormous effort to create suitable
dimensionality reduction methods .

Hence, we investigate a practical and efficient tool to describe and
analyze complex dynamic systems, namely symbolic dynamics . We derive
symbolic dynamic systems by representing the phase spaces as partitions
endowed with the Markov property, called Markov partitions. Such phase
space discretizations facilitate analyzing and computing global
dynamics. Further, they make dynamic systems accessible to optimal
sequential decision-making algorithms. Thus, we extend the work of to
bridge a substantial gap in research work by developing algorithms to
construct Markov partitions for various dynamic systems automatically.
Further, we formalize and implement a framework to build Markov decision
processes for dynamic systems based on Markov partitions. We show how to
fuse our work with approximate dynamic programming and apply this fused
framework in experiments executed in dynamic system environments.

Moreover, our experiments with Markov partitions provide evidence for
superior convergence performance of iterative Monte Carlo based policy
evaluation algorithms compared to regular grid-like discretizations.
However, we also experience mathematically inherent limits while
constructing Markov partitions for a broad class of dynamic systems.
Additionally, it is not clear how to entirely transport Markov
partitions’ topological and measure-theoretic properties beyond the
application to some fixed policy evaluation step.

### Project Structure

- [experimental/notebooks](https://github.com/juliusrueckin/masters-thesis/tree/master/experimental/notebooks) contains jupyter notebooks providing code for applying implemented algorithms, toy examples and performed experiments
    * _torus-toy-example-construction.ipynb_ contains the implementation of the construction work done in chapter 4.1
    * _markov-partition-construction-algorithms.py.ipynb_ contains the algorithmic constructions done in chapter 4.2
    * _monte-carlo-experiments.ipynb_ contains the Monte Carlo estimation algorithms experiment results evaluated in chapter 6.1
    * _dynamic-programming-experiments.ipynb_ contains the DP-algorithm experiment results evaluated in chapter 6.2
- [experimental/utils](https://github.com/juliusrueckin/masters-thesis/tree/master/experimental/utils) contains wrappers for frequently used code and proposed algorithms
    * _dynamic_system.py_ is a wrapper for defining system dynamics and computing (hyperbolic) fixed points
    * _partition.py_ is an implementation of the proposed Markov partition construction alogrithm in chapter chapter 4.2 given a dynamic system
    * _markov_decision_process.py_ is a parallelized implementation of the proposed Monte Carlo estimation algorithms in chapter 5 given a Markov partition
- [deployment/](https://github.com/juliusrueckin/masters-thesis/tree/master/deployment) contains everything required for a docker-compose setup of the whole pipeline

### Thesis Structure

1. Introduction
2. On Symbolic Dynamics for Representations of Dynamic Systems
    1. Shift Spaces
    2. Shifts of Finite Type
    3. From Dynamic Systems to Shifts of Finite Type
3. From Dynamic Systems to Markov Partitions and Symbolic Dynamics
    1. Topological Universality of Toral Endomorphisms
    2. Locally Split Anosov-Smale Hyperbolic Systems
    3. From Differentiable to Combinatorial Systems
    4. On Markov Partitions to Construct Symbolic Systems
4. Algorithmic Construction of Markov Partitions
    1. Implementation of Markov Partitions for a Toy Exmaple
    2. Algorithmic Construction of Markov Partitions
5. From Markov Partitions to Markov Decision Processes
6. Experiments and Applications
    1. Experiments - Monte Carlo Algorithms
    2. Experiments - Dynamic Programming Algorithms
    3. Discussion of Real-World Applications
7. Related Work
    1. Origins in the Theory of Dynamic Systems
    2. Symbolic Dynamics and Markov Partitions
    3. Representation Learning in Dynamic Programming
8. Conclusion and Future Work

### Execution of Pipeline

1. `cd deployment/`
2. `docker-compose build`
3. `docker-compose up`
4. Open jupyterlab link printed in terminal in a browser of your choice and start exploring the notebooks

### Presentation of Thesis Results

An extended presentation of the theoretical, algorithmical and experimental work carried out in this thesis can be found [over here.](https://docs.google.com/presentation/d/1pbK4_ac4LFXKli0opt8VTar8cc7b4QRxTLHrpihlT-k/edit?usp=sharing)
