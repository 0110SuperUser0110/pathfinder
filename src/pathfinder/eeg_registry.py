from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .runtime import external_root


@dataclass(slots=True)
class DependencyRequirement:
    import_name: str
    package_name: str | None = None
    purpose: str = ""

    @property
    def display_name(self) -> str:
        return self.package_name or self.import_name


@dataclass(slots=True)
class DependencyStatus:
    import_name: str
    package_name: str
    installed: bool
    purpose: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AssetRequirement:
    label: str
    path: Path
    required: bool = True
    description: str = ""


@dataclass(slots=True)
class AssetStatus:
    label: str
    path: Path
    exists: bool
    required: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


@dataclass(slots=True)
class ModelVariant:
    variant_id: str
    label: str
    assets: list[AssetRequirement] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class VariantStatus:
    variant_id: str
    label: str
    assets: list[AssetStatus]
    notes: str = ""

    @property
    def ready(self) -> bool:
        return all(asset.exists for asset in self.assets if asset.required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "label": self.label,
            "ready": self.ready,
            "notes": self.notes,
            "assets": [asset.to_dict() for asset in self.assets],
        }


@dataclass(slots=True)
class ModelStatus:
    model_id: str
    display_name: str
    repo_root: Path
    import_roots: list[Path]
    entrypoint: str
    input_contract: str
    output_contract: str
    dependencies: list[DependencyStatus]
    common_assets: list[AssetStatus]
    variants: list[VariantStatus]
    default_variant: str
    notes: list[str] = field(default_factory=list)

    @property
    def dependencies_ready(self) -> bool:
        return all(item.installed for item in self.dependencies)

    @property
    def common_assets_ready(self) -> bool:
        return all(item.exists for item in self.common_assets if item.required)

    @property
    def default_variant_ready(self) -> bool:
        for variant in self.variants:
            if variant.variant_id == self.default_variant:
                return variant.ready
        return False

    @property
    def runnable(self) -> bool:
        return self.dependencies_ready and self.common_assets_ready and self.default_variant_ready

    @property
    def missing_dependencies(self) -> list[str]:
        return [item.package_name for item in self.dependencies if not item.installed]

    @property
    def missing_assets(self) -> list[str]:
        missing = [str(item.path) for item in self.common_assets if item.required and not item.exists]
        for variant in self.variants:
            if variant.variant_id == self.default_variant:
                missing.extend(str(item.path) for item in variant.assets if item.required and not item.exists)
        return missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "repo_root": str(self.repo_root),
            "import_roots": [str(path) for path in self.import_roots],
            "entrypoint": self.entrypoint,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "dependencies_ready": self.dependencies_ready,
            "common_assets_ready": self.common_assets_ready,
            "default_variant_ready": self.default_variant_ready,
            "runnable": self.runnable,
            "default_variant": self.default_variant,
            "missing_dependencies": self.missing_dependencies,
            "missing_assets": self.missing_assets,
            "dependencies": [item.to_dict() for item in self.dependencies],
            "common_assets": [item.to_dict() for item in self.common_assets],
            "variants": [variant.to_dict() for variant in self.variants],
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class LoadedModel:
    model_id: str
    display_name: str
    variant_id: str
    device: str
    model: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelLoadError(RuntimeError):
    pass


@contextmanager
def prepend_sys_path(paths: list[Path]) -> Iterator[None]:
    original = list(sys.path)
    try:
        for path in reversed(paths):
            sys.path.insert(0, str(path))
        yield
    finally:
        sys.path[:] = original


class BaseModelAdapter:
    model_id: str = ""
    display_name: str = ""
    entrypoint: str = ""
    default_variant_id: str = ""

    def repo_root(self) -> Path:
        raise NotImplementedError

    def import_roots(self) -> list[Path]:
        return [self.repo_root()]

    def dependency_requirements(self) -> list[DependencyRequirement]:
        return []

    def common_asset_requirements(self) -> list[AssetRequirement]:
        return [
            AssetRequirement(
                label="repo_root",
                path=self.repo_root(),
                required=True,
                description="Local clone of the upstream repository.",
            )
        ]

    def variants(self) -> list[ModelVariant]:
        raise NotImplementedError

    def input_contract(self) -> str:
        return ""

    def output_contract(self) -> str:
        return ""

    def notes(self) -> list[str]:
        return []

    def status(self) -> ModelStatus:
        dependencies = [
            DependencyStatus(
                import_name=item.import_name,
                package_name=item.display_name,
                installed=importlib.util.find_spec(item.import_name) is not None,
                purpose=item.purpose,
            )
            for item in self.dependency_requirements()
        ]
        common_assets = [
            AssetStatus(
                label=item.label,
                path=item.path,
                exists=item.path.exists(),
                required=item.required,
                description=item.description,
            )
            for item in self.common_asset_requirements()
        ]
        variant_statuses = []
        for variant in self.variants():
            variant_statuses.append(
                VariantStatus(
                    variant_id=variant.variant_id,
                    label=variant.label,
                    notes=variant.notes,
                    assets=[
                        AssetStatus(
                            label=item.label,
                            path=item.path,
                            exists=item.path.exists(),
                            required=item.required,
                            description=item.description,
                        )
                        for item in variant.assets
                    ],
                )
            )
        return ModelStatus(
            model_id=self.model_id,
            display_name=self.display_name,
            repo_root=self.repo_root(),
            import_roots=self.import_roots(),
            entrypoint=self.entrypoint,
            input_contract=self.input_contract(),
            output_contract=self.output_contract(),
            dependencies=dependencies,
            common_assets=common_assets,
            variants=variant_statuses,
            default_variant=self.default_variant_id,
            notes=self.notes(),
        )

    def load(self, variant_id: str | None = None, device: str = "cpu") -> LoadedModel:
        variant_id = variant_id or self.default_variant_id
        status = self.status()
        if status.missing_dependencies:
            raise ModelLoadError(
                f"{self.model_id} is missing Python dependencies: {', '.join(status.missing_dependencies)}"
            )
        if not status.common_assets_ready:
            raise ModelLoadError(
                f"{self.model_id} is missing required assets: {', '.join(str(item.path) for item in status.common_assets if item.required and not item.exists)}"
            )
        variant = self._variant_by_id(variant_id)
        missing_variant_assets = [str(item.path) for item in variant.assets if item.required and not item.path.exists()]
        if missing_variant_assets:
            raise ModelLoadError(
                f"{self.model_id} variant {variant_id} is missing required assets: {', '.join(missing_variant_assets)}"
            )
        return self._load_impl(variant_id=variant_id, device=device)

    def _variant_by_id(self, variant_id: str) -> ModelVariant:
        for variant in self.variants():
            if variant.variant_id == variant_id:
                return variant
        raise ModelLoadError(f"unknown variant {variant_id!r} for model {self.model_id}")

    def _load_impl(self, variant_id: str, device: str) -> LoadedModel:
        raise NotImplementedError


class EEGPTAdapter(BaseModelAdapter):
    model_id = "eegpt"
    display_name = "EEGPT"
    entrypoint = "Modules.models.EEGPT_mcae:EEGTransformer"
    default_variant_id = "large_mixed_58ch_4s"

    CANONICAL_CHANNELS = [
        "FP1", "FPZ", "FP2",
        "AF3", "AF4",
        "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
        "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
        "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
        "PO7", "PO3", "POZ", "PO4", "PO8",
        "O1", "OZ", "O2",
    ]

    def repo_root(self) -> Path:
        return external_root() / "EEGPT"

    def import_roots(self) -> list[Path]:
        return [self.repo_root() / "downstream"]

    def dependency_requirements(self) -> list[DependencyRequirement]:
        return [
            DependencyRequirement("torch", "torch", "Load the pretrained checkpoint and backbone."),
            DependencyRequirement("numpy", "numpy", "Numeric preprocessing used by the upstream code."),
        ]

    def variants(self) -> list[ModelVariant]:
        return [
            ModelVariant(
                variant_id=self.default_variant_id,
                label="Large mixed-dataset 58-channel 4-second checkpoint",
                assets=[
                    AssetRequirement(
                        label="checkpoint",
                        path=self.repo_root() / "checkpoint" / "eegpt_mcae_58chs_4s_large4E.ckpt",
                        required=True,
                        description="Official pretrained EEGPT checkpoint.",
                    )
                ],
                notes="Expects the canonical 58-channel ordering used in the upstream downstream scripts.",
            )
        ]

    def input_contract(self) -> str:
        return "Tensor[B, 58, T] in the EEGPT canonical channel order; 4-second 256 Hz windows are the closest match to the released checkpoint."

    def output_contract(self) -> str:
        return "Feature tensor emitted by the pretrained EEGTransformer backbone."

    def notes(self) -> list[str]:
        return [
            "The adapter loads the pretrained target encoder rather than a downstream classifier head.",
            "Channel remapping should happen before invocation if the study montage differs from the canonical 58-channel order.",
        ]

    def _load_impl(self, variant_id: str, device: str) -> LoadedModel:
        with prepend_sys_path(self.import_roots()):
            torch = importlib.import_module("torch")
            nn = importlib.import_module("torch.nn")
            from functools import partial
            module = importlib.import_module("Modules.models.EEGPT_mcae")
            EEGTransformer = getattr(module, "EEGTransformer")

            checkpoint_path = self._variant_by_id(variant_id).assets[0].path
            backbone = EEGTransformer(
                img_size=[len(self.CANONICAL_CHANNELS), int(2.1 * 256)],
                patch_size=64,
                patch_stride=32,
                embed_num=4,
                embed_dim=512,
                depth=8,
                num_heads=8,
                mlp_ratio=4.0,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                init_std=0.02,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
            target_encoder_state = {
                key[len("target_encoder.") :]: value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith("target_encoder.")
            }
            backbone.load_state_dict(target_encoder_state, strict=False)
            channel_ids = backbone.prepare_chan_ids(self.CANONICAL_CHANNELS)
            backbone = backbone.to(device)
            backbone.eval()
            return LoadedModel(
                model_id=self.model_id,
                display_name=self.display_name,
                variant_id=variant_id,
                device=device,
                model=backbone,
                metadata={
                    "channel_names": list(self.CANONICAL_CHANNELS),
                    "channel_ids": channel_ids,
                    "checkpoint_path": str(checkpoint_path),
                },
            )


class EEGMambaAdapter(BaseModelAdapter):
    model_id = "eegmamba"
    display_name = "EEGMamba"
    entrypoint = "models.eegmamba:EEGMamba"
    default_variant_id = "foundation"

    def repo_root(self) -> Path:
        return external_root() / "EEGMamba"

    def dependency_requirements(self) -> list[DependencyRequirement]:
        return [
            DependencyRequirement("torch", "torch", "Load the pretrained checkpoint and backbone."),
            DependencyRequirement("einops", "einops", "Tensor reshaping used by the upstream backbone."),
            DependencyRequirement("mamba_ssm", "mamba-ssm", "Core Mamba layers required by EEGMamba."),
        ]

    def variants(self) -> list[ModelVariant]:
        return [
            ModelVariant(
                variant_id=self.default_variant_id,
                label="Foundation checkpoint",
                assets=[
                    AssetRequirement(
                        label="checkpoint",
                        path=self.repo_root() / "pretrained_weights" / "pretrained_weights.pth",
                        required=True,
                        description="Official EEGMamba pretrained checkpoint.",
                    )
                ],
                notes="The upstream quick start uses 200-point patches and returns per-patch embeddings.",
            )
        ]

    def input_contract(self) -> str:
        return "Tensor[B, C, L, P] where P defaults to 200 points per patch; raw Tensor[B, C, T] should be patched before inference."

    def output_contract(self) -> str:
        return "Tensor[B, C, L, D] of per-patch embeddings after the output projection layer."

    def notes(self) -> list[str]:
        return [
            "The released backbone depends on the external mamba-ssm package.",
            "Pathfinder should patch continuous time series into 200-point windows before invocation.",
        ]

    def _load_impl(self, variant_id: str, device: str) -> LoadedModel:
        with prepend_sys_path(self.import_roots()):
            torch = importlib.import_module("torch")
            module = importlib.import_module("models.eegmamba")
            EEGMamba = getattr(module, "EEGMamba")
            checkpoint_path = self._variant_by_id(variant_id).assets[0].path
            backbone = EEGMamba()
            backbone.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
            backbone = backbone.to(device)
            backbone.eval()
            return LoadedModel(
                model_id=self.model_id,
                display_name=self.display_name,
                variant_id=variant_id,
                device=device,
                model=backbone,
                metadata={"checkpoint_path": str(checkpoint_path), "patch_size": 200},
            )


class BrainOmniAdapter(BaseModelAdapter):
    model_id = "brainomni"
    display_name = "BrainOmni"
    entrypoint = "brainomni.model:BrainOmni"
    default_variant_id = "tiny"

    def repo_root(self) -> Path:
        return external_root() / "BrainOmni"

    def dependency_requirements(self) -> list[DependencyRequirement]:
        return [
            DependencyRequirement("torch", "torch", "Load checkpoints and run the backbone."),
            DependencyRequirement("einops", "einops", "Tensor reshaping used throughout BrainOmni."),
            DependencyRequirement("einx", "einx", "Used by the tokenizer quantization stack."),
            DependencyRequirement("vector_quantize_pytorch", "vector-quantize-pytorch", "Tokenizer codebook implementation."),
            DependencyRequirement("deepspeed", "deepspeed", "Imported by the upstream quantizer module even for local loading."),
        ]

    def common_asset_requirements(self) -> list[AssetRequirement]:
        common = super().common_asset_requirements()
        common.extend(
            [
                AssetRequirement(
                    label="tokenizer_checkpoint",
                    path=self.repo_root() / "ckpt_collection" / "braintokenizer" / "BrainTokenizer.pt",
                    required=True,
                    description="Released BrainTokenizer checkpoint.",
                ),
                AssetRequirement(
                    label="tokenizer_config",
                    path=self.repo_root() / "ckpt_collection" / "braintokenizer" / "model_cfg.json",
                    required=True,
                    description="Released BrainTokenizer config.",
                ),
            ]
        )
        return common

    def variants(self) -> list[ModelVariant]:
        return [
            ModelVariant(
                variant_id="tiny",
                label="BrainOmni tiny",
                assets=[
                    AssetRequirement(
                        label="checkpoint",
                        path=self.repo_root() / "ckpt_collection" / "tiny" / "BrainOmni.pt",
                        required=True,
                        description="Released BrainOmni tiny checkpoint.",
                    ),
                    AssetRequirement(
                        label="config",
                        path=self.repo_root() / "ckpt_collection" / "tiny" / "model_cfg.json",
                        required=True,
                        description="Released BrainOmni tiny config.",
                    ),
                ],
                notes="Best default for local experimentation because it is materially smaller than the base model.",
            ),
            ModelVariant(
                variant_id="base",
                label="BrainOmni base",
                assets=[
                    AssetRequirement(
                        label="checkpoint",
                        path=self.repo_root() / "ckpt_collection" / "base" / "BrainOmni.pt",
                        required=False,
                        description="Released BrainOmni base checkpoint.",
                    ),
                    AssetRequirement(
                        label="config",
                        path=self.repo_root() / "ckpt_collection" / "base" / "model_cfg.json",
                        required=False,
                        description="Released BrainOmni base config.",
                    ),
                ],
                notes="Heavier checkpoint intended for stronger representations if local hardware permits.",
            ),
        ]

    def input_contract(self) -> str:
        return "Tensor x[B, C, T] plus sensor positions pos[B, C, 6] and sensor_type[B, C]."

    def output_contract(self) -> str:
        return "Normalized feature tensor from BrainOmni.encode with shape approximately [B, C, W, D]."

    def notes(self) -> list[str]:
        return [
            "The released BrainOmni checkpoints include a tokenizer stack and require sensor metadata at inference time.",
            "Pathfinder will need a montage-to-position translation layer before this adapter can run against arbitrary studies.",
        ]

    def _load_impl(self, variant_id: str, device: str) -> LoadedModel:
        with prepend_sys_path(self.import_roots()):
            torch = importlib.import_module("torch")
            module = importlib.import_module("brainomni.model")
            BrainOmni = getattr(module, "BrainOmni")
            variant = self._variant_by_id(variant_id)
            config_path = next(item.path for item in variant.assets if item.label == "config")
            checkpoint_path = next(item.path for item in variant.assets if item.label == "checkpoint")
            model_config = json.loads(config_path.read_text(encoding="utf-8"))
            backbone = BrainOmni(**model_config)
            checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
            backbone.load_state_dict(checkpoint, strict=False)
            for parameter in backbone.tokenizer.parameters():
                parameter.requires_grad = False
            backbone = backbone.to(device)
            backbone.eval()
            return LoadedModel(
                model_id=self.model_id,
                display_name=self.display_name,
                variant_id=variant_id,
                device=device,
                model=backbone,
                metadata={"checkpoint_path": str(checkpoint_path), "config_path": str(config_path)},
            )


class BIOTAdapter(BaseModelAdapter):
    model_id = "biot"
    display_name = "BIOT"
    entrypoint = "model.biot:BIOTEncoder"
    default_variant_id = "six_datasets_18ch"

    def repo_root(self) -> Path:
        return external_root() / "BIOT"

    def dependency_requirements(self) -> list[DependencyRequirement]:
        return [
            DependencyRequirement("torch", "torch", "Load checkpoints and run the encoder."),
            DependencyRequirement("numpy", "numpy", "Used by the upstream encoder perturbation path."),
            DependencyRequirement("linear_attention_transformer", "linear-attention-transformer", "Core transformer block for BIOT."),
            DependencyRequirement("einops", "einops", "Used by some downstream utilities and wrappers."),
        ]

    def variants(self) -> list[ModelVariant]:
        return [
            ModelVariant(
                variant_id="prest_16ch",
                label="EEG PREST 16-channel",
                assets=[
                    AssetRequirement(
                        label="checkpoint",
                        path=self.repo_root() / "pretrained-models" / "EEG-PREST-16-channels.ckpt",
                        required=False,
                        description="PREST pretraining checkpoint with 16 channels.",
                    )
                ],
                notes="Best match for 16-channel montages.",
            ),
            ModelVariant(
                variant_id="shhs_prest_18ch",
                label="SHHS + PREST 18-channel",
                assets=[
                    AssetRequirement(
                        label="checkpoint",
                        path=self.repo_root() / "pretrained-models" / "EEG-SHHS+PREST-18-channels.ckpt",
                        required=False,
                        description="18-channel checkpoint pretrained on SHHS and PREST.",
                    )
                ],
                notes="Useful when a study can be normalized to the released 18-channel montage.",
            ),
            ModelVariant(
                variant_id="six_datasets_18ch",
                label="Six datasets 18-channel",
                assets=[
                    AssetRequirement(
                        label="checkpoint",
                        path=self.repo_root() / "pretrained-models" / "EEG-six-datasets-18-channels.ckpt",
                        required=True,
                        description="18-channel checkpoint pretrained across six EEG datasets.",
                    )
                ],
                notes="Default release because it covers the broadest source mix.",
            ),
        ]

    def input_contract(self) -> str:
        return "Tensor[B, C, T] of continuous EEG samples. The released checkpoints target either 16-channel or 18-channel montages."

    def output_contract(self) -> str:
        return "Embedding tensor[B, 256] from the BIOT encoder."

    def notes(self) -> list[str]:
        return [
            "The adapter loads the encoder instead of a task-specific classifier head.",
            "Choose the checkpoint variant that matches the channel count used after Pathfinder montage normalization.",
        ]

    def _load_impl(self, variant_id: str, device: str) -> LoadedModel:
        with prepend_sys_path(self.import_roots()):
            torch = importlib.import_module("torch")
            module = importlib.import_module("model.biot")
            BIOTEncoder = getattr(module, "BIOTEncoder")
            checkpoint_path = self._variant_by_id(variant_id).assets[0].path
            n_channels = 16 if variant_id == "prest_16ch" else 18
            encoder = BIOTEncoder(
                emb_size=256,
                heads=8,
                depth=4,
                n_channels=n_channels,
                n_fft=200,
                hop_length=100,
            )
            encoder.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
            encoder = encoder.to(device)
            encoder.eval()
            return LoadedModel(
                model_id=self.model_id,
                display_name=self.display_name,
                variant_id=variant_id,
                device=device,
                model=encoder,
                metadata={"checkpoint_path": str(checkpoint_path), "n_channels": n_channels},
            )



class CBraModAdapter(BaseModelAdapter):
    model_id = "cbramod"
    display_name = "CBraMod"
    entrypoint = "models.cbramod:CBraMod"
    default_variant_id = "foundation"

    def repo_root(self) -> Path:
        return external_root() / "CBraMod"

    def dependency_requirements(self) -> list[DependencyRequirement]:
        return [
            DependencyRequirement("torch", "torch", "Load checkpoints and run the backbone."),
            DependencyRequirement("einops", "einops", "Used by downstream quick-start utilities and tensor helpers."),
        ]

    def variants(self) -> list[ModelVariant]:
        return [
            ModelVariant(
                variant_id=self.default_variant_id,
                label="Foundation checkpoint",
                assets=[
                    AssetRequirement(
                        label="checkpoint",
                        path=self.repo_root() / "pretrained_weights" / "pretrained_weights.pth",
                        required=True,
                        description="Official CBraMod pretrained checkpoint.",
                    )
                ],
                notes="Windows-friendly transformer baseline that fills the fourth runnable slot when EEGMamba cannot be built natively.",
            )
        ]

    def input_contract(self) -> str:
        return "Tensor[B, C, L, P] where P defaults to 200 points per patch."

    def output_contract(self) -> str:
        return "Tensor[B, C, L, D] of per-patch embeddings after the CBraMod projection layer."

    def notes(self) -> list[str]:
        return [
            "CBraMod is used here as the Windows-compatible fourth local EEG engine.",
            "Its input contract closely matches EEGMamba, which keeps Pathfinder's patching logic consistent.",
        ]

    def _load_impl(self, variant_id: str, device: str) -> LoadedModel:
        with prepend_sys_path(self.import_roots()):
            torch = importlib.import_module("torch")
            module = importlib.import_module("models.cbramod")
            CBraMod = getattr(module, "CBraMod")
            checkpoint_path = self._variant_by_id(variant_id).assets[0].path
            backbone = CBraMod()
            backbone.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
            backbone = backbone.to(device)
            backbone.eval()
            return LoadedModel(
                model_id=self.model_id,
                display_name=self.display_name,
                variant_id=variant_id,
                device=device,
                model=backbone,
                metadata={"checkpoint_path": str(checkpoint_path), "patch_size": 200},
            )
class ModelRegistry:
    def __init__(self, adapters: list[BaseModelAdapter]) -> None:
        self._adapters = {adapter.model_id: adapter for adapter in adapters}

    def model_ids(self) -> list[str]:
        return sorted(self._adapters)

    def all(self) -> list[BaseModelAdapter]:
        return [self._adapters[key] for key in self.model_ids()]

    def get(self, model_id: str) -> BaseModelAdapter:
        normalized = model_id.strip().lower()
        if normalized not in self._adapters:
            raise KeyError(f"unknown model_id {model_id!r}")
        return self._adapters[normalized]

    def statuses(self) -> list[ModelStatus]:
        return [adapter.status() for adapter in self.all()]

    def load(self, model_id: str, variant_id: str | None = None, device: str = "cpu") -> LoadedModel:
        return self.get(model_id).load(variant_id=variant_id, device=device)


_DEFAULT_REGISTRY: ModelRegistry | None = None


def build_default_registry() -> ModelRegistry:
    return ModelRegistry(
        adapters=[
            EEGPTAdapter(),
            EEGMambaAdapter(),
            BrainOmniAdapter(),
            BIOTAdapter(),
            CBraModAdapter(),
        ]
    )


def default_registry() -> ModelRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = build_default_registry()
    return _DEFAULT_REGISTRY


