import numpy as np
import pyscf

class integral_provider:

	def __init__(self, molecule_object) -> None:
		"""Initialize integral class.
		Args:
		molecule_object_: Molecule class object.
		"""


		self.molecule_object = molecule_object


		# Run RHF:
		mf = pyscf.scf.RHF(self.molecule_object)
		mf.verbose = 0
		mf.kernel()

		# Get MO:
		self.c_mo = mf.mo_coeff


	def overlap_matrix(self) -> np.ndarray:
		"""Compute overlap integral matrix.
		Returns:
		Overlap integral matrix.
		"""

		S_AO = self.molecule_object.intor("int1e_ovlp")
		S_MO = np.einsum('uj,uv,vi->ij', self.c_mo, S_AO, self.c_mo)

		return S_MO


	def orbital_angular_momentum(self) -> np.ndarray:
		"""Compute orbital angular momentum.
	    Returns:
	        orbital angular momentum.
	    """

		L_x_MO, L_y_MO, L_z_MO = np.einsum('uj,xuv,vi->xij',  self.c_mo, self.molecule_object.intor("int1e_cg_irxp", comp=3),  self.c_mo)

		return L_x_MO, L_y_MO, L_z_MO

