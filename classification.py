"""Classification algorithms for hyperspectral chemical detection."""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter


class BandRestrictor:
    """Restricts spectral data to selected bands."""
    
    def __init__(self, selected_band_mask: np.ndarray):
        self.selected_band_mask = selected_band_mask
        self.selected_indices = np.where(selected_band_mask)[0]
        self.n_selected = len(self.selected_indices)
        self.reduction_ratio = self.n_selected / len(selected_band_mask)
    
    def restrict(
        self,
        refined_vector: np.ndarray,
        training_data: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if refined_vector.shape[0] != len(self.selected_band_mask):
            raise ValueError(f"Vector size mismatch: {refined_vector.shape[0]} vs {len(self.selected_band_mask)}")
        
        reduced_vector = refined_vector[self.selected_indices]
        
        if training_data is not None:
            if training_data.shape[1] != len(self.selected_band_mask):
                raise ValueError(f"Training data band mismatch")
            reduced_training_data = training_data[:, self.selected_indices]
        else:
            reduced_training_data = None
        
        return reduced_vector, reduced_training_data
    
    def restrict_batch(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors[:, self.selected_indices]
    
    def get_selected_wavelengths(self, wavelengths: np.ndarray) -> np.ndarray:
        return wavelengths[self.selected_indices]


class MultiLabelKNNPredictor:
    """Multi-label KNN classifier using binary relevance."""
    
    def __init__(
        self,
        k_neighbors: int = 5,
        distance_metric: str = 'euclidean',
        weight_neighbors: str = 'uniform',
    ):
        self.k_neighbors = k_neighbors
        self.distance_metric = distance_metric
        self.weight_neighbors = weight_neighbors
        self.classifiers: List[KNeighborsClassifier] = []
        self.n_chemicals: int = 0
        self.class_distributions: List[dict] = []
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        
        self.n_chemicals = y_train.shape[1]
        self.classifiers = []
        self.class_distributions = []
        
        for chem_idx in range(self.n_chemicals):
            y_single = y_train[:, chem_idx]
            
            pos_count = np.sum(y_single)
            neg_count = len(y_single) - pos_count
            self.class_distributions.append({
                'positive': pos_count,
                'negative': neg_count,
                'ratio': pos_count / len(y_single) if len(y_single) > 0 else 0.0
            })
            
            if pos_count == 0 or neg_count == 0:
                self.classifiers.append(None)
                continue
            
            classifier = KNeighborsClassifier(
                n_neighbors=min(self.k_neighbors, len(y_single)),
                metric=self.distance_metric,
                weights=self.weight_neighbors,
            )
            classifier.fit(X_train, y_single)
            self.classifiers.append(classifier)
    
    def predict(
        self,
        reduced_sample: np.ndarray,
        reduced_training_data: np.ndarray,
        training_labels: np.ndarray,
    ) -> np.ndarray:
        distances = np.linalg.norm(reduced_training_data - reduced_sample, axis=1)
        k_actual = min(self.k_neighbors, len(distances))
        nearest_indices = np.argsort(distances)[:k_actual]
        nearest_distances = distances[nearest_indices]
        
        if training_labels.ndim == 1:
            training_labels = training_labels.reshape(-1, 1)
        
        chemical_vector = np.zeros(training_labels.shape[1], dtype=int)
        
        for chem_idx in range(training_labels.shape[1]):
            neighbor_labels = training_labels[nearest_indices, chem_idx]
            
            if self.weight_neighbors == 'distance':
                weights = 1.0 / (nearest_distances + 1e-8)
                weights /= weights.sum()
                vote = np.sum(neighbor_labels * weights)
            else:
                vote = np.mean(neighbor_labels)
            
            chemical_vector[chem_idx] = int(vote >= 0.5)
        
        return chemical_vector
    
    def predict_fitted(self, reduced_sample: np.ndarray) -> np.ndarray:
        if reduced_sample.ndim == 1:
            reduced_sample = reduced_sample.reshape(1, -1)
        
        chemical_vector = np.zeros(self.n_chemicals, dtype=int)
        
        for chem_idx, classifier in enumerate(self.classifiers):
            if classifier is None:
                chemical_vector[chem_idx] = 0
            else:
                prediction = classifier.predict(reduced_sample)
                chemical_vector[chem_idx] = int(prediction[0])
        
        return chemical_vector
    
    def predict_proba_fitted(self, reduced_sample: np.ndarray) -> np.ndarray:
        if reduced_sample.ndim == 1:
            reduced_sample = reduced_sample.reshape(1, -1)
        
        probabilities = np.zeros(self.n_chemicals)
        
        for chem_idx, classifier in enumerate(self.classifiers):
            if classifier is None:
                probabilities[chem_idx] = 0.0
            else:
                proba = classifier.predict_proba(reduced_sample)
                probabilities[chem_idx] = proba[0, 1] if proba.shape[1] > 1 else 0.0
        
        return probabilities


class ResultDecoder:
    """Decodes multi-hot vectors to chemical names."""
    
    def __init__(self, chemical_names: List[str]):
        self.chemical_names = chemical_names
        self.n_chemicals = len(chemical_names)
        self.name_to_idx = {name: idx for idx, name in enumerate(chemical_names)}
    
    def decode(self, chemical_vector: np.ndarray) -> List[str]:
        if len(chemical_vector) != self.n_chemicals:
            raise ValueError(f"Vector size mismatch: {len(chemical_vector)} vs {self.n_chemicals}")
        
        detected_chemicals = [
            self.chemical_names[idx]
            for idx, is_present in enumerate(chemical_vector)
            if is_present == 1
        ]
        
        return detected_chemicals
    
    def encode(self, chemical_names: List[str]) -> np.ndarray:
        chemical_vector = np.zeros(self.n_chemicals, dtype=int)
        for name in chemical_names:
            if name in self.name_to_idx:
                chemical_vector[self.name_to_idx[name]] = 1
        return chemical_vector
    
    def decode_with_confidence(
        self,
        chemical_vector: np.ndarray,
        probabilities: np.ndarray,
    ) -> List[Tuple[str, float]]:
        results = [
            (self.chemical_names[idx], float(probabilities[idx]))
            for idx, is_present in enumerate(chemical_vector)
            if is_present == 1
        ]
        return sorted(results, key=lambda x: x[1], reverse=True)


def identify_chemicals(
    refined_vector: np.ndarray,
    training_data: np.ndarray,
    training_labels: np.ndarray,
    selected_band_mask: np.ndarray,
    chemical_names: List[str],
    k_neighbors: int = 5,
    return_probabilities: bool = False,
) -> Tuple[List[str], Optional[np.ndarray]]:
    restrictor = BandRestrictor(selected_band_mask)
    reduced_vector, reduced_training = restrictor.restrict(refined_vector, training_data)
    
    predictor = MultiLabelKNNPredictor(k_neighbors=k_neighbors)
    chemical_vector = predictor.predict(reduced_vector, reduced_training, training_labels)
    
    decoder = ResultDecoder(chemical_names)
    detected_chemicals = decoder.decode(chemical_vector)
    
    if return_probabilities:
        predictor.fit(reduced_training, training_labels)
        probabilities = predictor.predict_proba_fitted(reduced_vector)
        return detected_chemicals, probabilities
    
    return detected_chemicals, None


class ChemicalIdentificationPipeline:
    """Complete pipeline for chemical identification."""
    
    def __init__(
        self,
        training_data: np.ndarray,
        training_labels: np.ndarray,
        selected_band_mask: np.ndarray,
        chemical_names: List[str],
        k_neighbors: int = 5,
    ):
        self.restrictor = BandRestrictor(selected_band_mask)
        self.predictor = MultiLabelKNNPredictor(k_neighbors=k_neighbors)
        self.decoder = ResultDecoder(chemical_names)
        
        reduced_training, _ = self.restrictor.restrict(training_data)
        self.predictor.fit(reduced_training, training_labels)
        
        self.is_fitted = True
    
    def predict(self, refined_vector: np.ndarray) -> List[str]:
        reduced_vector, _ = self.restrictor.restrict(refined_vector)
        chemical_vector = self.predictor.predict_fitted(reduced_vector)
        detected_chemicals = self.decoder.decode(chemical_vector)
        return detected_chemicals
    
    def predict_with_confidence(self, refined_vector: np.ndarray) -> List[Tuple[str, float]]:
        reduced_vector, _ = self.restrictor.restrict(refined_vector)
        probabilities = self.predictor.predict_proba_fitted(reduced_vector)
        chemical_vector = self.predictor.predict_fitted(reduced_vector)
        results = self.decoder.decode_with_confidence(chemical_vector, probabilities)
        return results
