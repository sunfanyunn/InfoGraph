import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from sklearn.utils import check_random_state

class Nystrom():
    
	def __init__(self, kernel, kernel_params=None, n_components=100, random_state=None):
		self.kernel = kernel
		self.kernel_params = kernel_params
		self.n_components = n_components
		self.random_state = random_state

	def fit(self, graphs, y=None):
		rnd = check_random_state(self.random_state)
		n_samples = len(graphs)

		# get basis vectors
		if self.n_components > n_samples:
			n_components = n_samples
		else:
			n_components = self.n_components
		n_components = min(n_samples, n_components)
		inds = rnd.permutation(n_samples)
		basis_inds = inds[:n_components]
		basis = []
		for ind in basis_inds:
			basis.append(graphs[ind])

		basis_kernel = self.kernel(basis, basis, **self._get_kernel_params())

		# sqrt of kernel matrix on basis vectors
		U, S, V = svd(basis_kernel)
		S = np.maximum(S, 1e-12)
		self.normalization_ = np.dot(U * 1. / np.sqrt(S), V)
		self.components_ = basis
		self.component_indices_ = inds
		return self

	def transform(self, graphs):
		embedded = self.kernel(graphs, self.components_, **self._get_kernel_params())
		return np.dot(embedded, self.normalization_.T)

	def _get_kernel_params(self):
		params = self.kernel_params
		if params is None:
		    params = {}
		    
		return params

