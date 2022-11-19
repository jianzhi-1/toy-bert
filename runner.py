# Model parameters
batch_size = 5  # Train batch size per core
d_model = 4  # model width
num_heads = 1  # Number of attention heads
num_layers = 1  # Number of transformer layers
dropout_rate = 0.1  # Dropout rate

learning_rate = 2e-4  # Max learning-rate
grad_clip_value = 0.25  # Gradient norm clip value

# Create the dataset.
train_dataset, vocab_size = getData(batch_size=batch_size)

# Set up the model, loss, and updater.
forward_fn = build_forward_fn(vocab_size, d_model, num_heads, num_layers, dropout_rate)
forward_fn = hk.transform(forward_fn)
loss_fn = functools.partial(lm_loss_fn, forward_fn.apply, vocab_size)

optimizer = optax.chain(
    optax.clip_by_global_norm(grad_clip_value),
    optax.adam(learning_rate, b1=0.9, b2=0.99))

updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

# Variables for visualization
loss_graph = []

print('Initializing parameters...')
rng = jax.random.PRNGKey(428)
data = train_dataset[0]
state = updater.init(rng, data)

print('Starting train loop...')
prev_time = time.time()

for i in range(1, len(train_dataset)):
  print("train_dataset {}".format(i))
  data = train_dataset[i]
  state, metrics = updater.update(state, data) # states are for Adam
  loss_graph.append(float(np.array(metrics['loss'])))
