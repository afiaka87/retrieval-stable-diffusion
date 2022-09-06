# Stable "Retrieval-Augmented" Diffusion (WIP)

Update: Didn't work - moving on.

This is an experiment of mine using `faiss` indices built with [clip-retrieval](https://github.com/afiaka87/clip-retrieval/branch/unclip)

This model was not originally trained with retrieval augmentation in mind, and so won't benefit as much from this.

Further, the latents produced the stable-diffusion autoencoder are hardly small. Storing more than ~100,000 can be a hassle. 
