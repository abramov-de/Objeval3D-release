"""The exact versions of the models and code the paper's numbers were produced with.

Everything that is pinned here was checked against the files the experiments actually used: the Hugging Face revision is the one in
the cache of the machine that ran them, and the SHA-256 of the weights is the one of that file.

    Depth Anything 3 (reconstruction)  code  github.com/ByteDance-Seed/Depth-Anything-3 @ DA3_CODE_COMMIT   (Apache-2.0)
                                       weights  huggingface.co/depth-anything/DA3-LARGE @ DA3_REVISION     (CC BY-NC 4.0)
    VGGT (reconstruction, second model) code  github.com/facebookresearch/vggt @ VGGT_CODE_COMMIT          (git submodule)
                                       weights  huggingface.co/facebook/VGGT-1B @ VGGT_REVISION            (CC BY-NC 4.0)

The segmentation heads are trained on the frozen backbone of DA3_REPO at DA3_REVISION and are not valid on any other weights; the
successor `DA3-LARGE-1.1` is a different network.
"""

DA3_REPO = "depth-anything/DA3-LARGE"
DA3_REVISION = "c54c26b16ec04d218e8d584ecf4bce082a9fcc20"
DA3_WEIGHTS_SHA256 = "eaf2ae06df55889ad23eb245c82e2dd2a30c0cbf7e3d873a118fa5ed27a3e421"  # model.safetensors
DA3_CODE_COMMIT = "a71fc79123a420678fff8a32034c339f328f44e9"

VGGT_REPO = "facebook/VGGT-1B"
VGGT_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
VGGT_WEIGHTS_SHA256 = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"  # model.safetensors
VGGT_CODE_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"

PROCESS_RES = 504  # long side of the images given to the reconstruction models
