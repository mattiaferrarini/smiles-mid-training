# Embeddings

This folder contains the module for the initialization of embeddings for new tokens.

### Strategies

- `default`: New tokens are initialized randomly according to the model's default configuration.
- `average`: Resizes the embedding layer and initializes the embeddings for all new tokens as the average of the pre-existing (base) embeddings.
- `elementwise`: Specific to hybrid chemical tokenizers. Decomposes chemical tokens (e.g., "NaCl") into their constituent elements (e.g., "Sodium", "Chlorine"), fetches the embeddings for these element names from the base model, and initializes the new token as the average of these element embeddings.