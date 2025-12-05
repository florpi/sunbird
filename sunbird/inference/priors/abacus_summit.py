from scipy.linalg.lapack import dpotrf, dpotri
import numpy as np


class AbacusSummitEllipsoid:

    def __init__(self):
        params = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']

        # center and bounds of the bounding ellipsoid
        self.prior_c, self.prior_A = self.bounding_ellipsoid(self.summit_cosmo_table(params))

    def invert_symmetric_positive_semidefinite_matrix(self, m):
        r"""Invert a symmetric positive sem-definite matrix. This function is
        faster than numpy.linalg.inv but does not work for arbitrary matrices.
        Attributes
        ----------
        m : numpy.ndarray
            Matrix to be inverted.
        Returns
        -------
        m_inv : numpy.ndarray
            Inverse of the matrix.
        """

        m_inv = dpotri(dpotrf(m, False, False)[0])[0]
        m_inv = np.triu(m_inv) + np.triu(m_inv, k=1).T
        return m_inv


    def bounding_ellipsoid(self, points, tol=0, max_iterations=1000):
        r"""Find an approximation to the minimum bounding ellipsoid to:math:`m`
        points in :math:`n`-dimensional space using the Khachiyan algorithm.
        This function is based on
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.116.7691 but
        implemented independently of the corresponding MATLAP implementation or
        other Python ports of the MATLAP code.
        Attributes
        ----------
        points : numpy.ndarray with shape (m, n)
            A 2-D array where each row represents a point.
        Returns
        -------
        c : numpy.ndarray
            The position of the ellipsoid.
        A : numpy.ndarray
            The bounds of the ellipsoid in matrix form, i.e.
            :math:`(x - c)^T A (x - c) \leq 1`.
        """

        m, n = points.shape
        q = np.append(points, np.ones(shape=(m, 1)), axis=1)
        u = np.repeat(1.0 / m, m)
        q_outer = np.array([np.outer(q_i, q_i) for q_i in q])
        e = np.diag(np.ones(m))

        for i in range(max_iterations):
            if i % 1000 == 0:
                v = np.einsum('ji,j,jk', q, u, q)
            g = np.einsum('ijk,jk', q_outer,
                        self.invert_symmetric_positive_semidefinite_matrix(v))
            j = np.argmax(g)
            d_u = e[j] - u
            a = (g[j] - (n + 1)) / ((n + 1) * (g[j] - 1))
            shift = np.linalg.norm(a * d_u)
            v = v * (1 - a) + a * q_outer[j]
            u = u + a * d_u
            if shift <= tol:
                break

        c = np.einsum('i,ij', u, points)
        A_inv = (np.einsum('ji,j,jk', points, u, points) - np.outer(c, c)) * n
        A = np.linalg.inv(A_inv)

        scale = np.amax(np.einsum('...i,ij,...j', points - c, A, points - c))
        A /= scale

        return c, A

    def summit_cosmo_table(self, params):
        import pandas
        table_fn = "/global/cfs/cdirs/desicollab/users/epaillas/code/sunbird/sunbird/inference/priors/summit_cosmologies.txt"
        df = pandas.read_csv(table_fn, delimiter=',')
        df.columns = df.columns.str.strip()
        df.columns = list(df.columns.str.strip('# ').values)
        return np.c_[[df[param] for param in params]].T

    def is_within(self, p):
        """
        Check if a point is within the ellipsoid.
        """
        return np.einsum('...i,ij,...j', p - self.prior_c, self.prior_A, p - self.prior_c) <= 1

    def log_likelihood(self, x):
        """
        Log likelihood function.
        """
        if self.is_within(x):
            return 0.0
        return -np.inf