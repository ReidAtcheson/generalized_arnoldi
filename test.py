import unittest
import arnoldi
import numpy as np
import scipy.linalg as la




class TestArnoldi(unittest.TestCase):
    #Test that basic arnoldi produces an orthogonal basis
    def test_arnoldi_orthogonal(self):
        rng = np.random.RandomState(0)
        A=rng.rand(10,10)
        v=rng.rand(10)
        k=5
        V,H = arnoldi.arnoldi(lambda x : A@x,v,k)
        I=np.eye(k+1)
        self.assertLess(la.norm(I-V.T@V),1e-14)


    #Test the factorization error of basic arnoldi
    def test_arnoldi(self):
        rng = np.random.RandomState(0)
        A=rng.rand(10,10)
        v=rng.rand(10)
        k=5
        V,H = arnoldi.arnoldi(lambda x : A@x,v,k)
        Avk = A@V[:,0:k]
        VH = V@H
        self.assertLess(la.norm(Avk-VH),1e-14)






unittest.main()
