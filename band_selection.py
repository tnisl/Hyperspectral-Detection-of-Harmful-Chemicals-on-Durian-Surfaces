"""Band selection using genetic algorithm for hyperspectral analysis."""

from __future__ import annotations

import numpy as np
from typing import Tuple, Callable, Optional, List, Dict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from dataclasses import dataclass
import warnings


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 50
    n_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.01
    elite_size: int = 5
    lambda_penalty: float = 0.1
    tournament_size: int = 3
    early_stop_generations: int = 15
    min_bands: int = 10
    max_bands_ratio: float = 0.7


class GeneticBandSelector:
    """Genetic algorithm for optimal spectral band selection."""
    
    def __init__(
        self,
        n_bands: int,
        config: Optional[GeneticConfig] = None,
        k_neighbors: int = 5,
        cv_folds: int = 3,
        random_state: Optional[int] = None,
    ):
        self.n_bands = n_bands
        self.config = config if config is not None else GeneticConfig()
        self.k_neighbors = k_neighbors
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
        self.best_mask: Optional[np.ndarray] = None
        self.best_fitness: float = -np.inf
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.generation_stats: List[Dict] = []
        self.convergence_generation: int = -1
    
    def select_bands(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        if X_train.shape[1] != self.n_bands:
            raise ValueError(f"Expected {self.n_bands} bands, got {X_train.shape[1]}")
        
        population = self._initialize_population()
        no_improvement_count = 0
        
        for generation in range(self.config.n_generations):
            fitness_scores = self._evaluate_population(population, X_train, y_train)
            
            diversity = self._compute_diversity(population)
            self.diversity_history.append(diversity)
            
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_mask = population[best_idx].copy()
                no_improvement_count = 0
                self.convergence_generation = generation
            else:
                no_improvement_count += 1
            
            self.fitness_history.append(self.best_fitness)
            self.generation_stats.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(fitness_scores),
                'diversity': diversity,
                'n_bands_best': np.sum(self.best_mask),
            })
            
            if no_improvement_count >= self.config.early_stop_generations:
                break
            
            parents = self._select_parents(population, fitness_scores)
            offspring = self._produce_offspring(parents)
            population = self._update_population(population, offspring, fitness_scores)
            
            if diversity < 0.05:
                population = self._inject_diversity(population)
        
        return self.best_mask
    
    def _evaluate_population(
        self,
        population: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> np.ndarray:
        fitness_scores = np.zeros(len(population))
        for i, mask in enumerate(population):
            fitness_scores[i] = self._evaluate_fitness(mask, X_train, y_train)
        return fitness_scores
    
    def _compute_diversity(self, population: np.ndarray) -> float:
        n_pop = len(population)
        if n_pop < 2:
            return 1.0
        
        distances = []
        for i in range(n_pop):
            for j in range(i + 1, n_pop):
                hamming_dist = np.sum(population[i] != population[j]) / self.n_bands
                distances.append(hamming_dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _inject_diversity(self, population: np.ndarray) -> np.ndarray:
        n_random = max(5, int(0.2 * len(population)))
        random_individuals = self._initialize_population()[:n_random]
        population[-n_random:] = random_individuals
        return population
    
    def _initialize_population(self) -> np.ndarray:
        min_bands = max(self.config.min_bands, 10)
        max_bands = int(self.n_bands * self.config.max_bands_ratio)
        
        population = np.zeros((self.config.population_size, self.n_bands), dtype=bool)
        
        for i in range(self.config.population_size):
            n_selected = self.rng.integers(min_bands, max_bands + 1)
            selected_indices = self.rng.choice(self.n_bands, size=n_selected, replace=False)
            population[i, selected_indices] = True
        
        return population
    
    def _evaluate_fitness(
        self,
        mask: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> float:
        n_selected = np.sum(mask)
        
        if n_selected < self.config.min_bands:
            return -np.inf
        if n_selected > int(self.n_bands * self.config.max_bands_ratio):
            return -np.inf
        
        X_reduced = X_train[:, mask]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            classifier = KNeighborsClassifier(n_neighbors=self.k_neighbors, n_jobs=-1)
            f1_scores = []
            
            for chem_idx in range(y_train.shape[1]):
                y_single = y_train[:, chem_idx]
                
                if np.sum(y_single) < 2 or np.sum(y_single) >= len(y_single) - 1:
                    continue
                
                try:
                    cv = min(self.cv_folds, np.sum(y_single))
                    if cv < 2:
                        f1_scores.append(0.0)
                        continue
                    
                    scores = cross_val_score(
                        classifier, X_reduced, y_single,
                        cv=cv, scoring='f1_macro', error_score=0.0
                    )
                    f1_scores.append(np.mean(scores))
                except Exception:
                    f1_scores.append(0.0)
            
            f1_macro = np.mean(f1_scores) if f1_scores else 0.0
            band_penalty = self.config.lambda_penalty * (n_selected / self.n_bands)
            fitness = f1_macro - band_penalty
            
            return float(fitness)
    
    def _select_parents(self, population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        parents = []
        pop_size = len(population)
        tournament_size = self.config.tournament_size
        
        for _ in range(pop_size):
            tournament_idx = self.rng.choice(pop_size, size=min(tournament_size, pop_size), replace=False)
            tournament_fitness = fitness_scores[tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        
        return np.array(parents)
    
    def _produce_offspring(self, parents: np.ndarray) -> np.ndarray:
        offspring = []
        pop_size = len(parents)
        
        for i in range(0, pop_size, 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % pop_size]
            
            if self.rng.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            offspring.extend([child1, child2])
        
        return np.array(offspring[:pop_size])
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        crossover_type = self.rng.choice(['single', 'uniform'])
        
        if crossover_type == 'uniform':
            mask = self.rng.random(self.n_bands) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
        else:
            point = self.rng.integers(1, self.n_bands)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
        
        return child1, child2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mutated = individual.copy()
        mutation_mask = self.rng.random(self.n_bands) < self.config.mutation_rate
        mutated[mutation_mask] = ~mutated[mutation_mask]
        
        n_selected = np.sum(mutated)
        if n_selected < self.config.min_bands:
            n_add = self.config.min_bands - n_selected
            inactive = np.where(~mutated)[0]
            if len(inactive) >= n_add:
                add_indices = self.rng.choice(inactive, size=n_add, replace=False)
                mutated[add_indices] = True
        
        return mutated
    
    def _update_population(
        self,
        population: np.ndarray,
        offspring: np.ndarray,
        fitness_scores: np.ndarray,
    ) -> np.ndarray:
        elite_size = self.config.elite_size
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        elites = population[elite_indices]
        new_population = np.vstack([elites, offspring[:-elite_size]])
        return new_population
    
    def get_statistics(self) -> Dict:
        return {
            'best_fitness': self.best_fitness,
            'n_bands_selected': int(np.sum(self.best_mask)) if self.best_mask is not None else 0,
            'convergence_generation': self.convergence_generation,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history,
            'generation_stats': self.generation_stats,
        }
