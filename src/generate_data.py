import numpy as np
import os

def generate_lognormal_data(size=100000, mu=0, sigma=1, seed=42, cache_file=None):
    """
    Generate reproducible log-normally distributed data with caching.
    
    Parameters:
    -----------
    size : int
        Number of samples to generate (default: 100,000)
    mu : float
        Mean of underlying normal distribution (default: 0)
    sigma : float
        Standard deviation of underlying normal distribution (default: 1)
    seed : int
        Random seed for reproducibility (default: 42)
    cache_file : str or None
        Path to cache file (default: auto-generated based on parameters)
    
    Returns:
    --------
    np.ndarray
        Array of log-normally distributed values
    """
    if cache_file is None:
        cache_file = f"data/lognormal_data_size{size}_mu{mu}_sigma{sigma}_seed{seed}.csv"
    
    if os.path.exists(cache_file):
        return np.loadtxt(cache_file, delimiter=',')
    
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    np.random.seed(seed)
    data = np.random.lognormal(mean=mu, sigma=sigma, size=size)
    
    np.savetxt(cache_file, data, delimiter=',')
    
    return data

def generate_multiple_datasets(num_datasets=1000, size=100000, mu=0, sigma=1, base_seed=42):
    """
    Generate multiple log-normal datasets with different seeds for robust benchmarking.
    
    Parameters:
    -----------
    num_datasets : int
        Number of datasets to generate (default: 1000)
    size : int
        Number of samples per dataset (default: 100,000)
    mu : float
        Mean of underlying normal distribution (default: 0)
    sigma : float
        Standard deviation of underlying normal distribution (default: 1)
    base_seed : int
        Base seed for reproducibility (default: 42)
    
    Returns:
    --------
    list of np.ndarray
        List of log-normally distributed datasets
    """
    datasets = []
    for i in range(num_datasets):
        data = generate_lognormal_data(size=size, mu=mu, sigma=sigma, seed=base_seed + i)
        datasets.append(data)
    return datasets
