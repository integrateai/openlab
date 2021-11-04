# Examples using FedJAX 

# Setup
`./setup.sh`

# Virtual environment setup
1. `poetry install`
2. [For GPU]: `pip install --upgrade jax jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html`

# Running Example
1. `poetry shell`
2. `python -m src.emnist`
