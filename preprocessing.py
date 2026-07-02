"""Preprocessing algorithms for hyperspectral chemical detection."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from scipy.signal import savgol_filter
from scipy.ndimage import binary_erosion, binary_dilation, label
from spectral.io import envi


class SpectralExtractor:
    """Extracts spectral signatures from hyperspectral images."""
    
    def __init__(
        self,
        intensity_threshold: float = 0.1,
        min_region_size: int = 1000,
        morphology_kernel_size: int = 5,
    ):
        self.intensity_threshold = intensity_threshold
        self.min_region_size = min_region_size
        self.kernel_size = morphology_kernel_size
        self.extraction_stats: dict = {}
    
    def extract(
        self,
        image_cubes: list[np.ndarray],
        wavelengths_list: list[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        spectral_vectors = []
        all_wavelengths = []
        self.extraction_stats = {'masks': [], 'pixel_counts': [], 'snr': []}
        
        for idx, (cube, wavelengths) in enumerate(zip(image_cubes, wavelengths_list)):
            if cube.shape[2] != len(wavelengths):
                raise ValueError(f"Cube {idx}: band count mismatch")
            
            mask = self._compute_foreground_mask(cube)
            mask = self._refine_mask(mask)
            
            if np.sum(mask) < self.min_region_size:
                raise ValueError(f"Cube {idx}: insufficient foreground pixels")
            
            mean_spectrum, std_spectrum = self._extract_spectra(cube, mask)
            snr = self._compute_snr(mean_spectrum, std_spectrum)
            
            spectral_vectors.append(mean_spectrum)
            all_wavelengths.append(wavelengths)
            self.extraction_stats['masks'].append(mask)
            self.extraction_stats['pixel_counts'].append(np.sum(mask))
            self.extraction_stats['snr'].append(snr)
        
        extracted_vector = np.concatenate(spectral_vectors)
        concatenated_wavelengths = np.concatenate(all_wavelengths)
        
        return extracted_vector, concatenated_wavelengths
    
    def _compute_foreground_mask(self, cube: np.ndarray) -> np.ndarray:
        intensity = np.mean(cube, axis=2)
        intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
        mask = intensity_norm > self.intensity_threshold
        return mask
    
    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=bool)
        mask = binary_erosion(mask, kernel)
        mask = binary_dilation(mask, kernel)
        labeled, num_features = label(mask)
        if num_features > 0:
            sizes = np.bincount(labeled.ravel())[1:]
            largest_label = np.argmax(sizes) + 1
            mask = labeled == largest_label
        return mask
    
    def _extract_spectra(self, cube: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        masked_spectra = cube[mask, :]
        mean_spectrum = np.mean(masked_spectra, axis=0)
        std_spectrum = np.std(masked_spectra, axis=0)
        return mean_spectrum, std_spectrum
    
    def _compute_snr(self, mean: np.ndarray, std: np.ndarray) -> float:
        snr = np.mean(mean / (std + 1e-8))
        return float(snr)
    
    def get_statistics(self) -> dict:
        return self.extraction_stats.copy()


class VectorNormalizer:
    """
    Implements Algorithm 2: Z-Score Normalization
    
    Standardizes spectral vectors using Z-score normalization to place
    all bands on a comparable scale.
    """
    
    def __init__(self):
        """Initialize the normalizer with storage for statistics."""
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted = False
    
    def fit(self, training_vectors: np.ndarray):
        """
        Compute normalization statistics from training data.
        
        Parameters
        ----------
        training_vectors : np.ndarray
            Training spectral vectors (n_samples, n_bands)
        """
        self.mean = np.mean(training_vectors, axis=0)
        self.std = np.std(training_vectors, axis=0)
        
        # Guard against zero standard deviation
        self.std = np.where(self.std == 0, 1.0, self.std)
        
        self.is_fitted = True
    
    def transform(self, vector: np.ndarray) -> np.ndarray:
        """
        Apply Z-score normalization to a spectral vector.
        
        Algorithm 2: Z-Score Normalization
        ----------------------------------
        Require: An extracted vector; per-band mean and standard deviation from training
        Ensure: A normalized vector
        
        1: subtract the per-band mean from the vector
        2: divide by the per-band standard deviation (guarded against zero)
        3: return the standardised vector
        
        Parameters
        ----------
        vector : np.ndarray
            Extracted spectral vector (n_bands,)
            
        Returns
        -------
        normalized_vector : np.ndarray
            Normalized spectral vector (n_bands,)
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        # Step 1: Subtract per-band mean
        centered = vector - self.mean
        
        # Step 2: Divide by per-band standard deviation
        normalized_vector = centered / self.std
        
        # Step 3: Return standardized vector
        return normalized_vector
    
    def fit_transform(self, training_vectors: np.ndarray) -> np.ndarray:
        """
        Fit normalizer and transform training vectors.
        
        Parameters
        ----------
        training_vectors : np.ndarray
            Training spectral vectors (n_samples, n_bands)
            
        Returns
        -------
        normalized_vectors : np.ndarray
            Normalized training vectors (n_samples, n_bands)
        """
        self.fit(training_vectors)
        
        # Apply normalization to each vector
        normalized_vectors = np.zeros_like(training_vectors)
        for i in range(training_vectors.shape[0]):
            normalized_vectors[i] = self.transform(training_vectors[i])
        
        return normalized_vectors


class VectorSmoother:
    """
    Implements Algorithm 3: Savitzky–Golay Smoothing
    
    Applies Savitzky-Golay filtering to remove noise while preserving
    spectral peaks and absorption features.
    """
    
    def __init__(self, window_length: int = 11, polyorder: int = 2):
        """
        Initialize the smoother.
        
        Parameters
        ----------
        window_length : int
            Length of the filter window (must be odd and > polyorder)
        polyorder : int
            Order of the polynomial used for fitting
        """
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length")
        
        self.window_length = window_length
        self.polyorder = polyorder
    
    def smooth(self, vector: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing to a spectral vector.
        
        Algorithm 3: Savitzky–Golay Smoothing
        -------------------------------------
        Require: A normalized vector; a window length and a polynomial degree
        Ensure: A refined vector
        
        1: apply the Savitzky–Golay filter to the vector
        2: return the smoothed vector
        
        Parameters
        ----------
        vector : np.ndarray
            Normalized spectral vector (n_bands,)
            
        Returns
        -------
        refined_vector : np.ndarray
            Smoothed spectral vector (n_bands,)
        """
        # Step 1: Apply Savitzky-Golay filter
        refined_vector = savgol_filter(
            vector,
            window_length=self.window_length,
            polyorder=self.polyorder,
            mode='nearest'
        )
        
        # Step 2: Return smoothed vector
        return refined_vector


def preprocess_sample(
    image_cubes: list[np.ndarray],
    wavelengths_list: list[np.ndarray],
    normalizer: VectorNormalizer,
    smoother: VectorSmoother,
) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single sample.
    
    Combines all three algorithms in sequence:
    1. Spectral Extraction (Algorithm 1)
    2. Z-Score Normalization (Algorithm 2)
    3. Savitzky-Golay Smoothing (Algorithm 3)
    
    Parameters
    ----------
    image_cubes : list[np.ndarray]
        List of hyperspectral image cubes
    wavelengths_list : list[np.ndarray]
        List of wavelength arrays
    normalizer : VectorNormalizer
        Fitted normalizer instance
    smoother : VectorSmoother
        Smoother instance
        
    Returns
    -------
    refined_vector : np.ndarray
        Fully preprocessed spectral vector
    """
    # Algorithm 1: Extract spectral vector
    extractor = SpectralExtractor()
    extracted_vector, _ = extractor.extract(image_cubes, wavelengths_list)
    
    # Algorithm 2: Normalize
    normalized_vector = normalizer.transform(extracted_vector)
    
    # Algorithm 3: Smooth
    refined_vector = smoother.smooth(normalized_vector)
    
    return refined_vector