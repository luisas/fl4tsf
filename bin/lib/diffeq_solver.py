import torch
import torch.nn as nn

# Corrected imports for torchode based on the provided library structure
from torchode import AutoDiffAdjoint, InitialValueProblem, ODETerm
from torchode.single_step_methods import Dopri5, Euler, Heun, Tsit5
from torchode.step_size_controllers import IntegralController


class DiffeqSolver(nn.Module):
    def __init__(self, input_dim, ode_func, method, latents, 
                 odeint_rtol=1e-3, odeint_atol=1e-4, device=torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.latents = latents
        self.device = device
        self.ode_func = ode_func
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        
        # Instantiate solver components. The term is omitted here so it can be
        # provided dynamically in the forward pass, which is more flexible.
        self.step_method = self._create_solver(method)
        self.step_size_controller = IntegralController(atol=self.odeint_atol, rtol=self.odeint_rtol)
        self.solver = AutoDiffAdjoint(self.step_method, self.step_size_controller)

    def _create_solver(self, method):
        """Map method strings to TorchODE single-step method objects"""
        # Mappings based on available solvers in the provided torchode library
        solver_map = {
            # Explicit methods
            'euler': Euler(term=None),
            'heun': Heun(term=None),
            'dopri5': Dopri5(term=None),
            'tsit5': Tsit5(term=None),
            
            # Backward compatibility mappings to available solvers
            'midpoint': Heun(term=None),
            'ralston': Heun(term=None),
        
        }
        
        if method in solver_map:
            return solver_map[method]
        else:
            # Default to Tsit5 for unknown methods as it's a robust choice
            print(f"Warning: Unknown method '{method}', defaulting to Tsit5")
            return Tsit5(term=None)

    def forward(self, first_point, time_steps_to_predict, backwards=False):
        """Decode trajectory through TorchODE solver"""
        n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
        n_dims = first_point.size()[-1]
        
        # Reshape for batch processing: combine traj_samples and traj dimensions
        batch_size = n_traj_samples * n_traj
        y0 = first_point.view(batch_size, n_dims)

        # Ensure time_steps_to_predict is 2D for the IVP
        if time_steps_to_predict.dim() == 1:
            time_steps_to_predict = time_steps_to_predict.unsqueeze(0).expand(batch_size, -1)
        
        # Create a wrapper function for the ODE that handles reshaping
        def batched_ode_func(t, y):
            # Reshape from [batch, features] to [n_traj_samples, n_traj, n_dims] for the model
            y_reshaped = y.view(n_traj_samples, n_traj, n_dims)
            
            # Call original ode function
            if backwards:
                dy_dt = -self.ode_func(t, y_reshaped)
            else:
                dy_dt = self.ode_func(t, y_reshaped)
            
            # Reshape back to [batch, features] for the solver
            return dy_dt.view(batch_size, n_dims)
        
        # Create ODE Term and Initial Value Problem
        term = ODETerm(batched_ode_func)
        ivp = InitialValueProblem(y0=y0, t_eval=time_steps_to_predict)
        
        # Solve using the adjoint method
        solution = self.solver.solve(ivp, term=term)
        
        # Reshape solution back to the expected 4D format
        # solution.ys shape: [batch_size, n_timepoints, n_dims]
        pred_y = solution.ys.view(n_traj_samples, n_traj, len(time_steps_to_predict[0]), n_dims)
        
        # Validation checks
        assert(torch.mean(pred_y[:, :, 0, :] - first_point) < 0.001)
        assert(pred_y.size()[0] == n_traj_samples)
        assert(pred_y.size()[1] == n_traj)

        return pred_y

    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, n_traj_samples=1):
        """Sample trajectory using prior with TorchODE"""
        
        # Get the sampling function from ode_func
        func = self.ode_func.sample_next_point_from_prior
        
        # Create a wrapper for the batched sampling function
        def batched_sample_func(t, y):
            return func(t, y)

        batch_size = starting_point_enc.shape[0]
        # Ensure time_steps_to_predict is 2D for the IVP
        if time_steps_to_predict.dim() == 1:
            time_steps_to_predict = time_steps_to_predict.unsqueeze(0).expand(batch_size, -1)
        
        # Create ODE Term and IVP for sampling
        term = ODETerm(batched_sample_func)
        ivp = InitialValueProblem(
            y0=starting_point_enc,
            t_eval=time_steps_to_predict
        )
        
        # Solve
        solution = self.solver.solve(ivp, term=term)
        
        # Return in [batch, time, features] format
        return solution.ys