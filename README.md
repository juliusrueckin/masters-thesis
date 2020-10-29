# On the Combination of Symbolic Dynamics and Dynamic Programming for Dynamic Systems

## Master’s Thesis - TU Munich - Chair of Data Processing

### Abstract

Investigation and understanding of nonlinear dynamic systems is crucial in a widespread variety of applications and fields of research, e.g.  describing chaotic systems  in  physics  [31],  comprehend  bio-molecular  processes  for  designing  newmedial drugs [4] or in engineering as for developing new high performance materials [51].  At the same time, the global dynamics of such systems are notoriouslyhard  to  represent,  although  differential  equations  to  describe  such  systems  arein some cases well-known.   A common tool for analyzing dynamic behavior overtime are dynamic simulations [19].  However, these techniques often suffer from avery high dimensional data space resulting in an enormous effort to create suitabledimensionality reduction methods [30] [48].

Hence,  we investigate a generally applicable and efficient tool to describe andanalyze  complex  dynamic  systems,  namely  symbolic  dynamics  [42].   We  derivesuch  symbolic  dynamic  systems  by  representing  the  phase  spaces  as  partitionsendowed with the Markov property, called Markov partitions. Such phase space discretizations are not only useful to analyze global dynamics, but also make the broadclass of dynamic systems accessible to algorithmic treatment of optimal sequentialdecision making problems in such environments. Thus, we bridge a substantial gapin research work of symbolic dynamics and develop algorithms to construct suchMarkov partitions in an automated fashion for various dynamic systems.  Further,we formalize and examine a framework to build Markov decision processes (MDP)for dynamic systems based on Markov partitions.  We show how to fuse our workwith the rich tool of approximate dynamic programming (ADP) and apply this fusedframework in experiments executed in dynamic system environments.

In this thesis, we successfully fuse the branches symbolic dynamics and ADP resulting in a mathematically well-defined method to derive MDPs relying on the statespace representations given by Markov partitions.  At its heart, we develop an algorithmic method to construct such Markov partitions in a computer by extendingresearch work of [22].  Moreover, our experiments give evidence for similar performance in some ADP settings for dynamic systems compared to regular state spacediscretizations. However, we also experience mathematically inherit limits while constructing Markov partitions for a broad class of dynamic systems.  Further, it is notyet clear how to fully transport the desirable mathematical properties of Markov partitions beyond the application to some fixed policy evaluation step.

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
