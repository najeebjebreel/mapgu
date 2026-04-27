from .kanon import mdav_clusters, probabilistic_k_anonymize_by_permutation

__all__ = [
    "mdav_clusters",
    "probabilistic_k_anonymize_by_permutation",
    "generate_dp_adult",
    "generate_dp_laplace_only",
    "generate_dp_cifar10_dppix",
    "dp_pix",
    "DPCIFAR10Dataset",
]


def __getattr__(name):
    if name in {"generate_dp_adult", "generate_dp_laplace_only", "generate_dp_cifar10_dppix"}:
        from .dp import generate_dp_adult, generate_dp_laplace_only, generate_dp_cifar10_dppix

        return {
            "generate_dp_adult": generate_dp_adult,
            "generate_dp_laplace_only": generate_dp_laplace_only,
            "generate_dp_cifar10_dppix": generate_dp_cifar10_dppix,
        }[name]

    if name == "dp_pix":
        from .dp_pix import dp_pix

        return dp_pix

    if name == "DPCIFAR10Dataset":
        from .cifar_dataset import DPCIFAR10Dataset

        return DPCIFAR10Dataset

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
