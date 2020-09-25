# On the Combination of SymbolicDynamics and Dynamic Programming for Dynamic Systems

## Master’s Thesis - TU Munich - Chair of Data Processing

### Abstract

Investigation and understanding of nonlinear dynamic systems is crucial in a widespread variety of applications and fields of research, e.g.  describing chaotic sys-tems in physics [13], comprehend bio-molecular processes for designing new me-dial drugs [3], in engineering as for developing new high performance materials [18]or even in computer science with applications for encryption algorithms [5].  At thesame time, the dynamics of such systems evolving over time are notoriously hardto represent, although differential equations to describe such systems are in somecases well-known. A common tool for analyzing dynamic behavior over time are dy-namic simulations [3]. Nevertheless, these techniques often suffer from a very highdimensional data space resulting in an enormous effort to create suitable dimen-sionality reduction methods helping researchers to find Markov state models (MSM)describing the system’s dynamics over time in an aggregated fashion [12] [16].Hence,  we investigate mathematical more convenient methods to describe dy-namic systems and their markovian dynamics.  A promising branch of mathemat-ics, called symbolic dynamics [14], evolved a few decades ago, unites rigorous andwell-studied ideas of algebraic topology analyzing topological spaces and endomor-phisms on such spaces with naturally arising and comprehensible methods of au-tomata and graph theory. Thus, symbolic dynamics provides a generally applicableand at the same time mathematical well-studied framework to describe dynamic sys-tems without suffering from tedious engineering work trying to approximate Markovstate models by complex dimensionality reduction methods.Moreover, deriving such Markov models for dynamic systems is not only useful toaggregate dynamics over time, but also makes the broad class of dynamic systemseasily  accessible  for  algorithmic  treatment  of  optimal  sequential  decision  makingproblems.  Being able to construct such Markov state models in a rigorous fashionfor dynamic systems allows for applications of approximate dynamic programming(ADP) algorithms solving such optimal control problems in an efficient way [6].Thus, in this thesis, we aim for an extensive study of the fusion of symbolic dy-namics and ADP, such that the mathematical generality of both frameworks lead toan overall combined methodology making nonlinear dynamic systems accessible tothe powerful class of ADP algorithms.  This approach circumvents the problem ofloosing mathematical exactness, e.g. by hand-engineered dimensionality reductiontechniques, but rather leverages symbolic dynamics as a rigorous mathematical toolto precisely describe nonlinear dynamic systems for ADP applications.

### Project Structure

- [experimental/notebooks](https://github.com/juliusrueckin/masters-thesis/tree/master/experimental/notebooks) contains jupyter notebooks providing code for applying proposed algorithms, implemented toy examples and executed experiments
- [experimental/utils](https://github.com/juliusrueckin/masters-thesis/tree/master/experimental/utils) contains wrappers for frequently used code, proposed algorithmic computations such as hyperbolic fixed point computations and developed algorithms for constructions of Markov partitions
- [deployment/](https://github.com/juliusrueckin/masters-thesis/tree/master/deployment) contains everything needed for a docker-compose setup of the whole pipeline

### Thesis Structure

1. Problem Statement
2. From Symbolic Dynamics to Representations of Dynamic Systems
    1. Shift Spaces
    2. Shifts of Finite Type
    3. Sofic Shifts
    4. From Dynamic Systems to Markov Partitions and Shifts of Finite Type
3. From Differential Geometry to Markov Partitions and Symbolic Dynamics
    1. Topological Universality of Toral Endomorphisms
    2. Geodesic Flows and Geodesic Universality
    3. Locally Split Anosov-Smale Hyperbolic Systems
    4. On Differential Geometry and Symbolic Dynamics
    5. On Differential Geometry and Markov Partitions
4. Construction of MDP Representations for Dynamic Systems
    1. Implementation of Markov Partitions for a Toy Exmaple
    2. Algorithmic Construction of Markov Partitions
    3. From Markov Partitions to Markov Decision Processes
5. Related Work

### Execution of Pipeline

1. `cd deployment/`
2. `docker-compose build`
3. `docker-compose up`
4. Open jupyterlab link printed in terminal in a browser of your choice and start exploring the notebooks
