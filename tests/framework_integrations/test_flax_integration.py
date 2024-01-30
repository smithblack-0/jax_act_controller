import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import unittest
from flax.training import train_state
from src import jax_act

class SumParametersLayer(nn.Module):
    num_params: int

    @nn.compact
    def __call__(self):
        params = self.param('params', nn.initializers.uniform(), (self.num_params,), jnp.float32)
        return jnp.sum(params)


class SimpleACTModel:
    def make_probabilities(self) -> jnp.ndarray:
        return 0.1 * jnp.ones(list(state.shape))

    def make_controller(self, state: jnp.ndarray):
        builder = jax_act.ControllerBuilder.new_builder(list(state.shape))
        builder = builder.define_accumulator_by_shape("output", list(state.shape))
        return builder.build()
    def run_layer(self, controller: jax_act.ACT_Controller, state: jnp.ndarray):
        output = self.layer()
        probabilities = self.make_probabilities()

        controller = controller.cache_update("output", output)
        controller = controller.iterate_act(probabilities)
        return controller, state


    def __init__(self, layer: SumParametersLayer):
        super().__init__()
        self.layer = layer

class SimpleTestModel(nn.Module):
    num_parameters: int

    @nn.compact
    def __call__(self, state)->jnp.ndarray:
        parameter_layer = SumParametersLayer(self.num_parameters)
        act_model = SimpleACTModel(parameter_layer)
        controller, state = jax_act.execute_act(act_model, state)
        return controller.accumulators["output"]


# Initialize the model and optimizer
model = SimpleTestModel(num_parameters=5)
params = model.init(jax.random.PRNGKey(0), jnp.array(0.0))
optimizer = optax.adam(learning_rate=0.1)

# Create a TrainState
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params['params'],
    tx=optimizer,
)

# Loss function
def loss_fn(params, dummy_input):
    return model.apply({'params': params}, dummy_input)

# Training step
@jax.jit
def train_step(state, dummy_input):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, dummy_input)
    return state.apply_gradients(grads=grads), loss

# Training loop
for step in range(100):
    state, loss = train_step(state, jnp.array(0.0))
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss}")

# Final sum of parameters
final_sum = model.apply({'params': state.params}, jnp.array(0.0))
print(f"Final sum of parameters: {final_sum}")
