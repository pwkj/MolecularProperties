import pyscf

import molecularintegrals.integrals_1e as integrals_1e


mol = pyscf.gto.Mole()
#mol.atom = 'H 0 0 0; H 0 0 0.74'
mol.atom = 'H 0 0 0'
mol.basis = "cc-pvdz"
mol.charge = 0
mol.spin = 1 #spin is the number of unpaired electrons 2S, i.e. the difference between the number of alpha and beta electrons.
mol.build()

ints_1e = integrals_1e.integral_provider(molecule_object = mol)
ints_1e.overlap_matrix()
ints_1e.orbital_angular_momentum()

print(ints_1e.overlap_matrix())
print(ints_1e.orbital_angular_momentum())