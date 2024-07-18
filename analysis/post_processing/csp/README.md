# Constraint Satisfaction for PID and Primary Prediction

## I. Usage

A *constraint* $C$ on some variable $X$ limits the possible values that $X$ can 
assume in its domain. For example, suppose we have a `Particle` instance `emshower` that have `semantic_label == 1`:
```python
print(emshower.semantic_type)
1
```
Let's make a constraint `ParticleSemanticConstraint`

Usually, we want to restrict a Particle's type and primary label based on heuristics that are well-grounded in physics. 

```

```