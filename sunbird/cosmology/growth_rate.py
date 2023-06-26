from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import optimize
from cosmoprimo.fiducial import AbacusSummit
from cosmoprimo.fiducial import DESI
import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
from flax.core import freeze

import optax


from sunbird.emulators.models.simple_flax import SimpleNN

DEFAULT_PATH = Path(__file__).parent.parent.parent
# Omega_m = (w_b + w_c + w_nu) / h^2
# https://github.com/cosmodesi/cosmoprimo/blob/6cb2850432f760f3100fb8fe1256c6a9175282eb/cosmoprimo/eisenstein_hu.py#L139


class Growth:
    def __init__(
        self,
        theta_star: float = AbacusSummit(0).theta_star,
        emulate=False,
        emulator_data_dir=DEFAULT_PATH / "data/hemu/",
    ):
        self.theta_star = theta_star
        self.emulate = emulate
        self.emulator_data_dir = emulator_data_dir
        if self.emulate:
            self.model = SimpleNN()
            self.params = freeze(
                np.load(
                    self.emulator_data_dir / "model_params.npy", allow_pickle=True
                ).item()
            )

    def generate_emulator_training_data(
        self,
        n_samples: int = 32_000,
    ):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        parameters = {
            "omega_b": [0.0207, 0.0243],
            "omega_cdm": [0.1032, 0.140],
            "sigma8": [0.678, 0.938],
            "N_ur": [1.188, 2.889],
            "n_s": [0.9012, 1.025],
            "w0_fld": [-1.22, -0.726],
            "wa_fld": [-0.628, 0.621],
        }
        samples = [
            np.random.uniform(low, high, n_samples // size)
            for low, high in parameters.values()
        ]
        samples_matrix = np.stack(samples, axis=-1)
        h_values, sample_parameters = [], []
        for i, sample in enumerate(samples_matrix):
            try:
                cosmology = self.get_cosmology_fixed_theta_star(
                    DESI(engine="class"),
                    dict(
                        theta_star=self.theta_star,
                        omega_b=sample[0],
                        omega_cdm=sample[1],
                        sigma8=sample[2],
                        N_ur=sample[3],
                        n_s=sample[4],
                        w0_fld=sample[5],
                        wa_fld=sample[6],
                    ),
                )
                sample_parameters.append(sample)
                h_values.append(cosmology.h)
            except:
                continue
        h_values = comm.gather(h_values, root=0)
        samples = comm.gather(sample_parameters, root=0)
        # skip_indices = comm.gather(skip_indices, root=0)
        if rank == 0:
            # samples_matrix = np.concatenate(all_samples,axis=0)
            h_values = [item for sublist in h_values for item in sublist]
            samples = np.array([item for sublist in samples for item in sublist])
            print("h shape = ", len(h_values))
            print("samples shape = ", samples.shape)
            data_dict = dict(zip(parameters.keys(), samples.T))
            data_dict["h"] = np.array(h_values)
            for key in data_dict.keys():
                print(key, len(data_dict[key]))
            df = pd.DataFrame(data_dict)
            df.to_csv(self.emulator_data_dir / "h_training_data.csv", index=False)

    def train_emulator(
        self,
    ):
        def loss_fn(params, x, y):
            return jnp.mean((model.apply(params, x) - y) ** 2)

        def update(params, opt_state, x, y):
            loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state, loss

        data = pd.read_csv(self.emulator_data_dir / "h_training_data.csv")
        print(f"Training on {len(data)} samples")
        model = SimpleNN()
        key = jax.random.PRNGKey(0)
        params = model.init(key, jnp.ones((1, len(data.columns) - 1)))
        X = data[
            ["omega_b", "omega_cdm", "sigma8", "N_ur", "n_s", "w0_fld", "wa_fld"]
        ].values
        y = data["h"].values[:, None]

        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(params)
        # Training loop
        epochs = 1500
        for epoch in range(epochs):
            params, opt_state, loss = update(params, opt_state, X, y)
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        print("Final MSE = ", loss)
        np.save(self.emulator_data_dir / "model_params.npy", dict(params))

    def Omega_m0(self, omega_cdm, omega_b, h, omega_ncdm: float = 0.00064420):
        return (omega_cdm + omega_b + omega_ncdm) / h**2

    def Omega_m(self, omega_cdm, omega_b, h, omega_ncdm, w0_fld, wa_fld, z):
        Omega_m0 = self.Omega_m0(omega_cdm, omega_b, h, omega_ncdm)
        return (
            Omega_m0
            * (1 + z) ** 3
            / (
                Omega_m0 * (1 + z) ** 3
                + (1 - Omega_m0)
                * (1 + z) ** (3.0 * (1 + w0_fld + wa_fld))
                * np.exp(-3.0 * wa_fld * z / (1 + z))
            )
        )

    def approximate_growth_rate(
        self,
        omega_cdm,
        omega_b,
        h,
        omega_ncdm,
        w0_fld,
        wa_fld,
        z,
    ):
        """
        Approximation of growth rate.

        References
        ----------
        https://arxiv.org/abs/astro-ph/0507263
        """
        wz1 = w0_fld + (1.0 - 0.5) * wa_fld
        return self.Omega_m(omega_cdm, omega_b, h, omega_ncdm, w0_fld, wa_fld, z) ** (
            0.55 + 0.05 * (1 + wz1)
        )

    def load_emulator_parameters(
        self,
    ):
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=self.emulator_data_dir, target=None
        )
        return restored_state

    def get_emulated_h(self, omega_b, omega_cdm, sigma8, N_ur, n_s, w0_fld, wa_fld):
        x = jnp.vstack([omega_b, omega_cdm, sigma8, N_ur, n_s, w0_fld, wa_fld]).T
        return self.model.apply(self.params, x)

    def get_cosmology_fixed_theta_star(
        self,
        fiducial,
        params,
        h_limits=[0.4, 1.0],
        xtol=1.0e-6,
    ):
        theta = params.pop("theta_star", None)
        fiducial = fiducial.clone(base="input", **params)
        if theta is not None:
            if "h" in params:
                raise ValueError("Cannot provide both theta_star and h")

            def f(h):
                cosmo = fiducial.clone(base="input", h=h)
                return 100.0 * (theta - cosmo.get_thermodynamics().theta_star)

            rtol = xtol
            try:
                h = optimize.bisect(f, *h_limits, xtol=xtol, rtol=rtol, disp=True)
            except ValueError as exc:
                raise ValueError(
                    "Could not find proper h value in the interval that matches theta_star = {:.4f} with [f({:.3f}), f({:.3f})] = [{:.4f}, {:.4f}]".format(
                        theta, *h_limits, *list(map(f, h_limits))
                    )
                ) from exc
            cosmo = fiducial.clone(base="input", h=h)
        return cosmo

    def get_growth(
        self,
        omega_b: float,
        omega_cdm: float,
        sigma8: float,
        n_s: float,
        N_ur: float,
        w0_fld: float,
        wa_fld: float,
        z: float,
        omega_ncdm: float = 0.00064420,
    ):
        if self.emulate:
            h = self.get_emulated_h(
                omega_b=omega_b,
                omega_cdm=omega_cdm,
                sigma8=sigma8,
                N_ur=N_ur,
                n_s=n_s,
                w0_fld=w0_fld,
                wa_fld=wa_fld,
            ).squeeze()
            return self.approximate_growth_rate(
                omega_cdm=omega_cdm,
                omega_b=omega_b,
                h=h,
                w0_fld=w0_fld,
                wa_fld=wa_fld,
                omega_ncdm=omega_ncdm,
                z=z,
            )
            """
            cosmology = DESI(**dict(
                    omega_b=omega_b,
                    omega_cdm=omega_cdm,
                    sigma8=sigma8,
                    n_s=n_s,
                    N_ur=N_ur,
                    w0_fld=w0_fld,
                    wa_fld=wa_fld,
                    h=h[0],
                ))
            """
        else:
            cosmology = self.get_cosmology_fixed_theta_star(
                DESI(engine="class"),
                dict(
                    theta_star=self.theta_star,
                    omega_b=omega_b,
                    omega_cdm=omega_cdm,
                    sigma8=sigma8,
                    n_s=n_s,
                    N_ur=N_ur,
                    w0_fld=w0_fld,
                    wa_fld=wa_fld,
                ),
            )
            return cosmology.growth_rate(z)


if __name__ == "__main__":
    import time

    t0 = time.time()
    growth = Growth()
    # growth.generate_emulator_training_data()
    growth.train_emulator()
    print(f"It took {time.time() - t0} seconds")
